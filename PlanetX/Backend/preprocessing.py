
import numpy as np
import lightkurve as lk

from astropy.time import Time


INPUT_LENGTH = 2000

# Начальные настройки поиска.
MIN_PERIOD_DAYS = 0.5
MAX_PERIOD_DAYS = 30.0
PERIOD_GRID_SIZE = 5000

MIN_TRANSIT_SNR = 7.0
MIN_OBSERVED_TRANSITS = 2

# Возможные длительности транзита в днях
DURATION_GRID = np.array([
    0.5, 1, 1.5, 2, 3, 4, 6, 8, 12
]) / 24.0


def find_transit(time, flux):

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux)

    time = time[valid]
    flux = flux[valid]

    if len(time) < 100:
        return {
            "found": False,
            "reason": "Недостаточно точек наблюдения."
        }

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    baseline = time[-1] - time[0]

    if baseline < 2:
        return {
            "found": False,
            "reason": "Слишком короткий интервал наблюдений."
        }

    max_period = min(
        MAX_PERIOD_DAYS,
        baseline / MIN_OBSERVED_TRANSITS
    )

    if max_period <= MIN_PERIOD_DAYS:
        return {
            "found": False,
            "reason": "Недостаточная продолжительность наблюдений."
        }

    # Создаём кривую блеска Lightkurve.
    # Для поиска важны разности времён, поэтому используем
    # единый числовой формат времени в днях.
    light_curve = lk.LightCurve(
        time=Time(time, format="jd", scale="tdb"),
        flux=flux
    )

    # Удаляем нечисловые значения
    light_curve = light_curve.remove_nans()

    period_grid = np.linspace(
        MIN_PERIOD_DAYS,
        max_period,
        PERIOD_GRID_SIZE
    )

    try:
        bls = light_curve.to_periodogram(
            method="bls",
            period=period_grid,
            duration=DURATION_GRID
        )

        best_index = int(np.nanargmax(bls.power.value))

        period = float(bls.period[best_index].value)

        epoch_value = bls.transit_time[best_index]
        epoch = float(
            epoch_value.value
            if hasattr(epoch_value, "value")
            else epoch_value
        )

        duration = float(bls.duration[best_index].value)
        depth = float(bls.depth[best_index].value)

        depth_snr = float(bls.snr[best_index])

        observed_transits = baseline / period

        """
        best_index = int(np.nanargmax(bls.power.value))

        period = float(bls.period[best_index].value)

        epoch = float(
            bls.transit_time[best_index].value
        )

        duration = float(
            bls.duration[best_index].value
        )

        depth = float(
            bls.depth[best_index].value
        )

        depth_snr = float(
            bls.depth_snr[best_index].value
        )

        observed_transits = baseline / period
        """

        # Первичная проверка убедительности сигнала
        if not np.isfinite(depth_snr):
            depth_snr = 0.0

        if (
            depth_snr < MIN_TRANSIT_SNR or
            observed_transits < MIN_OBSERVED_TRANSITS or
            depth <= 0 or
            duration <= 0
        ):
            return {
                "found": False,
                "reason": "Не найден достаточно убедительный периодический транзит.",
                "best_period_days": period,
                "depth_snr": depth_snr
            }

        return {
            "found": True,
            "period_days": period,
            "epoch": epoch,
            "duration_days": duration,
            "depth": depth,
            "depth_snr": depth_snr,
            "observed_transits": observed_transits
        }

    except Exception as error:

        return {
            "found": False,
            "reason": f"Ошибка BLS: {str(error)}"
        }


def prepare_signal(time, flux, period, epoch, duration_days):

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux)

    time = time[valid]
    flux = flux[valid]

    if len(time) < 100:
        raise ValueError("Недостаточно точек для подготовки сигнала.")

    median_flux = np.median(flux)

    if not np.isfinite(median_flux) or median_flux == 0:
        raise ValueError("Некорректная медианная яркость.")

    # Такая же нормализация, как при обучении
    flux = flux / median_flux - 1.0

    # Фаза относительно найденной эпохи
    phase_days = (
        (time - epoch + period / 2) % period
    ) - period / 2

    phase = phase_days / duration_days

    # Участок вокруг транзита: от -5 до +5 длительностей
    mask = np.abs(phase) <= 5

    phase = phase[mask]
    flux = flux[mask]

    if len(phase) < 20:
        raise ValueError(
            "Недостаточно точек около предполагаемого транзита."
        )

    order = np.argsort(phase)
    phase = phase[order]
    flux = flux[order]

    unique_phase, inverse = np.unique(
        phase,
        return_inverse=True
    )

    summed_flux = np.bincount(
        inverse,
        weights=flux
    )

    counts = np.bincount(inverse)

    mean_flux = summed_flux / counts

    if len(unique_phase) < 20:
        raise ValueError("Недостаточно уникальных фаз.")

    # Не экстраполируем за пределы реально имеющихся данных
    if unique_phase[0] > -4.5 or unique_phase[-1] < 4.5:
        raise ValueError(
            "Недостаточное покрытие участка транзита."
        )

    target_phase = np.linspace(
        -5,
        5,
        INPUT_LENGTH
    )

    prepared = np.interp(
        target_phase,
        unique_phase,
        mean_flux
    )

    std = np.std(prepared)

    if not np.isfinite(std) or std < 1e-10:
        raise ValueError("Сигнал имеет слишком малое изменение яркости.")

    prepared = (prepared - np.mean(prepared)) / std

    return prepared.astype(np.float32)

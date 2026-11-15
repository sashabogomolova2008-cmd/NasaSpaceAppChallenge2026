// =================================
// PLANETX — FRONTEND LOGIC
// =================================

console.log("PLANETX SCRIPT LOADED!");

console.log("Analyze button:", document.getElementById("analyzeButton"));
console.log("File input:", document.getElementById("fileInput"));

const API_URL = "http://127.0.0.1:8000/analyze";

const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");

const chartPlaceholder = document.getElementById("chartPlaceholder");
const chartCanvas = document.getElementById("blsChart");

const resultTitle = document.getElementById("resultTitle");
const resultDescription = document.getElementById("resultDescription");
const resultIcon = document.getElementById("resultIcon");

const modelStatus = document.getElementById("modelStatus");
const cnnPlaceholder = document.getElementById("cnnPlaceholder");
const cnnResults = document.getElementById("cnnResults");

let selectedFile = null;


// =================================
// FILE SELECTION
// =================================

fileInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        showStatus("Please select a CSV file.", true);
        selectedFile = null;
        analyzeButton.disabled = true;
        fileName.textContent = "No file selected";
        return;
    }

    selectedFile = file;

    fileName.textContent = "Selected file: " + file.name;

    analyzeButton.disabled = false;

    showStatus("File selected. Ready for analysis.");

});


// =================================
// DRAG AND DROP
// =================================

const uploadArea = document.querySelector(".upload-area");

uploadArea.addEventListener("dragover", function (event) {
    event.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        showStatus("Please upload a CSV file.", true);
        return;
    }

    selectedFile = file;

    fileInput.files = event.dataTransfer.files;

    fileName.textContent = "Selected file: " + file.name;

    analyzeButton.disabled = false;

    showStatus("File selected. Ready for analysis.");

});


// =================================
// STATUS MESSAGE
// =================================

function showStatus(message, isError = false) {

    statusMessage.textContent = message;

    statusMessage.style.color = isError
        ? "#FF8295"
        : "#91A8C2";

}


// =================================
// ANALYSIS BUTTON
// =================================

analyzeButton.addEventListener("click", async function () {

    console.log("PLANETX: Analyze button clicked");

    if (!selectedFile) {
        showStatus("Please select a CSV file first.", true);
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    showStatus("Uploading data and running BLS analysis...");
    resetResults();

    try {

        console.log("PLANETX: POST", API_URL);

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        console.log("Server response status:", response.status);

        let data;

        try {
            data = await response.json();
        } catch (e) {
            throw new Error(
                "Server did not return JSON (HTTP " + response.status +
                "). Check that the backend is running at " + API_URL
            );
        }

        console.log("Server response data:", data);

        if (!response.ok) {
            throw new Error(
                data.detail || "Server returned an error."
            );
        }

        handleAnalysisResult(data);

    } catch (error) {

        console.error("PLANETX error:", error);

        showStatus("Analysis failed: " + error.message, true);

        resultTitle.textContent = "Analysis error";

        resultDescription.textContent =
            "Please check that the backend is running and your CSV is valid.";

        resultIcon.textContent = "!";

    } finally {

        analyzeButton.disabled = false;

        analyzeButton.innerHTML =
            'Analyze light curve <span>→</span>';

    }

});


// =================================
// RESET RESULTS
// =================================

function resetResults() {

    document.getElementById("periodValue").textContent = "—";
    document.getElementById("durationValue").textContent = "—";
    document.getElementById("depthValue").textContent = "—";
    document.getElementById("snrValue").textContent = "—";

    resultTitle.textContent = "Analyzing...";
    resultDescription.textContent = "Please wait for the BLS result.";
    resultIcon.textContent = "✧";

    cnnPlaceholder.classList.remove("hidden");
    cnnResults.classList.add("hidden");

    modelStatus.textContent = "WAITING FOR MODEL";

    chartCanvas.style.display = "none";
    chartPlaceholder.classList.remove("hidden");

}


// =================================
// HANDLE BACKEND RESPONSE
// =================================

function handleAnalysisResult(data) {

    if (data.status === "no_transit") {

        showStatus("BLS analysis completed.");

        resultTitle.textContent =
            "No convincing transit detected";

        resultDescription.textContent =
            data.message ||
            "No convincing periodic transit signal was found.";

        resultIcon.textContent = "○";

        modelStatus.textContent = "NOT RUN";

        cnnPlaceholder.classList.remove("hidden");
        cnnResults.classList.add("hidden");

        return;
    }


    if (data.status === "model_not_ready") {

        showBLSResults(data.bls);

        resultTitle.textContent =
            "Transit-like signal detected";

        resultDescription.textContent =
            "BLS found a periodic signal. The neural network is not installed yet.";

        resultIcon.textContent = "✧";

        modelStatus.textContent = "MODEL NOT CONNECTED";

        cnnPlaceholder.classList.remove("hidden");
        cnnResults.classList.add("hidden");

        showStatus("BLS completed. CNN is waiting for its trained model.");

        return;
    }


    if (data.status === "transit_found") {

        showBLSResults(data.bls);

        resultTitle.textContent =
            "Transit-like signal detected";

        resultDescription.textContent =
            "A periodic transit-like signal was found. CNN classification is shown below.";

        resultIcon.textContent = "✦";

        showCNNResults(data.neural_network);

        showStatus("Analysis completed successfully.");

        return;
    }


    throw new Error(
        data.message || "Unknown response from backend."
    );

}


// =================================
// BLS RESULTS
// =================================

function showBLSResults(bls) {

    if (!bls) {
        return;
    }

    document.getElementById("periodValue").textContent =
        formatNumber(bls.period);

    document.getElementById("durationValue").textContent =
        formatNumber(bls.duration);

    document.getElementById("depthValue").textContent =
        formatNumber(bls.depth);

    document.getElementById("snrValue").textContent =
        formatNumber(bls.snr);

    // Draw a real periodogram only if the backend supplies
    // the period and power arrays.

    if (
        Array.isArray(bls.periods) &&
        Array.isArray(bls.power) &&
        bls.periods.length === bls.power.length &&
        bls.periods.length > 1
    ) {

        drawBLSChart(bls.periods, bls.power);

    } else {

        chartCanvas.style.display = "none";

        chartPlaceholder.classList.remove("hidden");

        chartPlaceholder.querySelector("strong").textContent =
            "BLS signal detected";

        chartPlaceholder.querySelector("span").textContent =
            "The best period has been calculated. The backend currently does not return the full periodogram data.";

    }

}


// =================================
// CNN RESULTS
// =================================

function showCNNResults(network) {

    if (!network) {

        cnnPlaceholder.classList.remove("hidden");
        cnnResults.classList.add("hidden");

        return;
    }

    cnnPlaceholder.classList.add("hidden");
    cnnResults.classList.remove("hidden");

    modelStatus.textContent = "MODEL CONNECTED";

    const confirmed = Number(
        network.confirmed_probability
    );

    const falsePositive = Number(
        network.false_positive_probability
    );

    document.getElementById("confirmedScore").textContent =
        (confirmed * 100).toFixed(1) + "%";

    document.getElementById("falsePositiveScore").textContent =
        (falsePositive * 100).toFixed(1) + "%";

    document.getElementById("confirmedBar").style.width =
        Math.max(0, Math.min(100, confirmed * 100)) + "%";

    document.getElementById("falsePositiveBar").style.width =
        Math.max(0, Math.min(100, falsePositive * 100)) + "%";

    document.getElementById("cnnPrediction").textContent =
        network.predicted_class || "Unknown";

}


// =================================
// BLS PERIODGRAM DRAWING
// =================================

function drawBLSChart(periods, power) {

    const canvas = chartCanvas;
    const ctx = canvas.getContext("2d");

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    const dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    chartCanvas.style.display = "block";

    chartPlaceholder.classList.add("hidden");

    ctx.clearRect(0, 0, width, height);

    const padding = {
        left: 55,
        right: 25,
        top: 25,
        bottom: 45
    };

    const chartWidth =
        width - padding.left - padding.right;

    const chartHeight =
        height - padding.top - padding.bottom;

    const minPeriod = Math.min(...periods);
    const maxPeriod = Math.max(...periods);

    const minPower = Math.min(...power);
    const maxPower = Math.max(...power);

    const powerRange = maxPower - minPower || 1;
    const periodRange = maxPeriod - minPeriod || 1;

    // Grid

    ctx.strokeStyle = "rgba(145, 168, 194, 0.15)";
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {

        const y =
            padding.top + (chartHeight / 4) * i;

        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

    }

    // Periodogram line

    ctx.beginPath();

    ctx.strokeStyle = "#79C8FF";
    ctx.lineWidth = 2;

    periods.forEach((period, index) => {

        const x =
            padding.left +
            ((period - minPeriod) / periodRange) * chartWidth;

        const y =
            padding.top +
            chartHeight -
            ((power[index] - minPower) / powerRange) * chartHeight;

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

    });

    ctx.stroke();

    // Axis labels

    ctx.fillStyle = "#91A8C2";
    ctx.font = "11px Arial";

    ctx.textAlign = "center";

    ctx.fillText(
        "Period (days)",
        padding.left + chartWidth / 2,
        height - 10
    );

    ctx.save();

    ctx.translate(15, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);

    ctx.fillText("BLS Power", 0, 0);

    ctx.restore();

    ctx.textAlign = "left";

    ctx.fillText(
        minPeriod.toFixed(2),
        padding.left,
        height - 25
    );

    ctx.textAlign = "right";

    ctx.fillText(
        maxPeriod.toFixed(2),
        width - padding.right,
        height - 25
    );

}


// =================================
// NUMBER FORMATTING
// =================================

function formatNumber(value) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(5);

}


// =================================
// INITIAL STATE
// =================================

resetResults();

showStatus("PlanetX is ready. Upload a CSV light curve.");


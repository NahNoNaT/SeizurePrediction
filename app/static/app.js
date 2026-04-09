function initializeDeleteConfirmations() {
    const elements = document.querySelectorAll("[data-confirm]");
    elements.forEach((element) => {
        element.addEventListener("click", (event) => {
            const message = element.getAttribute("data-confirm") || "Are you sure?";
            if (!window.confirm(message)) {
                event.preventDefault();
            }
        });
    });
}

function initializeLoadingButtons() {
    const buttons = document.querySelectorAll("[data-loading-label]");
    buttons.forEach((button) => {
        const form = button.closest("form");
        if (!form) {
            return;
        }
        form.addEventListener("submit", () => {
            button.dataset.originalLabel = button.textContent;
            button.textContent = button.getAttribute("data-loading-label");
            button.disabled = true;
        });
    });
}

function initializeFileInputs() {
    const fileInputs = document.querySelectorAll("[data-file-input]");
    fileInputs.forEach((input) => {
        const container = input.closest(".upload-dropzone");
        const label = container ? container.querySelector("[data-file-name]") : null;
        if (!label) {
            return;
        }

        input.addEventListener("change", () => {
            const file = input.files && input.files[0];
            label.textContent = file ? file.name : "No file selected";
        });
    });
}

function buildChartGradient(context, colorStart, colorEnd) {
    const chart = context.chart;
    const { ctx, chartArea } = chart;
    if (!chartArea) {
        return colorEnd;
    }

    const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
}

function initializeRiskChart() {
    const canvas = document.getElementById("riskTimelineChart");
    if (!canvas || typeof Chart === "undefined") {
        return;
    }

    const rawPoints = canvas.getAttribute("data-chart-points");
    if (!rawPoints) {
        return;
    }

    let points = [];
    try {
        points = JSON.parse(rawPoints);
    } catch (error) {
        console.error("Unable to parse chart data", error);
        return;
    }

    if (!Array.isArray(points) || points.length === 0) {
        return;
    }

    const labels = points.map((point) => point.label);
    const values = points.map((point) => Number(point.value) * 100);
    const highlightedValues = points.map((point) => (point.highlight ? Number(point.value) * 100 : null));

    new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: canvas.getAttribute("data-chart-label") || "Estimated seizure risk",
                    data: values,
                    fill: true,
                    tension: 0.34,
                    borderColor: "#165e86",
                    backgroundColor(context) {
                        return buildChartGradient(context, "rgba(22, 94, 134, 0.22)", "rgba(22, 94, 134, 0.02)");
                    },
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointBackgroundColor: "#165e86",
                    borderWidth: 3,
                },
                {
                    label: "Flagged review period",
                    data: highlightedValues,
                    fill: false,
                    tension: 0.34,
                    borderColor: "#157d84",
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointBackgroundColor: "#157d84",
                    spanGaps: false,
                    borderWidth: 3,
                },
            ],
        },
        options: {
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: "top",
                    labels: {
                        usePointStyle: true,
                        boxWidth: 10,
                        color: "#1c3342",
                        font: {
                            family: "'Source Sans 3', sans-serif",
                            size: 13,
                            weight: "600",
                        },
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(16, 37, 50, 0.94)",
                    titleFont: {
                        family: "'Source Sans 3', sans-serif",
                        size: 13,
                        weight: "700",
                    },
                    bodyFont: {
                        family: "'Source Sans 3', sans-serif",
                        size: 13,
                    },
                    callbacks: {
                        label(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                        },
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: "#5f7887",
                        callback(value) {
                            return `${value}%`;
                        },
                    },
                    grid: {
                        color: "rgba(95, 120, 135, 0.16)",
                    },
                    border: {
                        display: false,
                    },
                },
                x: {
                    ticks: {
                        color: "#5f7887",
                        maxTicksLimit: 10,
                    },
                    grid: {
                        display: false,
                    },
                    border: {
                        display: false,
                    },
                },
            },
        },
    });
}

function setWaveformStatus(viewer, message, isError = false) {
    const status = viewer.querySelector("[data-waveform-status]");
    if (!status) {
        return;
    }

    status.textContent = message;
    status.classList.toggle("is-error", isError);
}

function setWaveformMessages(viewer, messages) {
    const container = viewer.querySelector("[data-waveform-messages]");
    if (!container) {
        return;
    }

    if (!Array.isArray(messages) || messages.length === 0) {
        container.hidden = true;
        container.innerHTML = "";
        return;
    }

    container.hidden = false;
    container.innerHTML = messages.map((message) => `<div class="message-item">${message}</div>`).join("");
}

function summarizeWaveformChannels(viewer, channels) {
    const summary = viewer.querySelector("[data-waveform-channel-summary]");
    if (!summary) {
        return;
    }

    if (!Array.isArray(channels) || channels.length === 0) {
        summary.textContent = "";
        return;
    }

    summary.textContent = `Displayed channels: ${channels.join(", ")}`;
}

function centerWaveformTrace(signal) {
    if (!Array.isArray(signal) || signal.length === 0) {
        return [];
    }

    const numericSignal = signal.map((value) => Number(value));
    const mean = numericSignal.reduce((sum, value) => sum + value, 0) / numericSignal.length;
    return numericSignal.map((value) => value - mean);
}

function escapeSvgText(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function renderWaveformPlot(viewer, payload) {
    const plot = viewer.querySelector("[data-waveform-plot]");
    if (!plot) {
        setWaveformStatus(viewer, "Waveform plotting is unavailable in this browser.", true);
        return;
    }

    if (!Array.isArray(payload.channels) || payload.channels.length === 0 || !Array.isArray(payload.times) || payload.times.length === 0) {
        clearWaveformCanvas(plot);
        setWaveformStatus(viewer, "No waveform samples were returned for the requested preview window.", true);
        summarizeWaveformChannels(viewer, []);
        return;
    }

    window.requestAnimationFrame(() => {
        drawWaveformCanvas(plot, payload);
        setWaveformStatus(
            viewer,
            `Showing ${payload.duration_sec.toFixed(1)}s from ${payload.start_sec.toFixed(1)}s at ${payload.sampling_rate.toFixed(1)} Hz.`,
        );
        summarizeWaveformChannels(viewer, payload.channels);
    });
}

function clearWaveformCanvas(plot) {
    if (!(plot instanceof HTMLElement)) {
        return;
    }

    if (plot instanceof HTMLCanvasElement) {
        const context = plot.getContext("2d");
        if (context) {
            context.clearRect(0, 0, plot.width, plot.height);
        }
    }

    plot.innerHTML = "";
}

function drawWaveformCanvas(plot, payload) {
    if (!(plot instanceof HTMLElement)) {
        return;
    }

    const width = Math.max(plot.clientWidth || plot.parentElement?.clientWidth || 640, 640);
    const channelCount = payload.channels.length;
    const height = Math.max(380, channelCount * 92);
    const margin = { top: 20, right: 20, bottom: 34, left: 84 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const timeStart = Number(payload.times[0] || 0);
    const timeEnd = Number(payload.times[payload.times.length - 1] || timeStart + 1);
    const timeSpan = Math.max(timeEnd - timeStart, 1e-6);
    const rowHeight = plotHeight / Math.max(channelCount, 1);
    const channels = payload.channels;
    const centeredSignals = payload.signals.map((signal) => centerWaveformTrace(signal));

    const parts = [
        `<svg class="waveform-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="EEG waveform preview" xmlns="http://www.w3.org/2000/svg">`,
        `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#ffffff" />`,
    ];

    for (let index = 0; index <= 6; index += 1) {
        const x = margin.left + (plotWidth * index) / 6;
        parts.push(
            `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}" stroke="rgba(95, 120, 135, 0.16)" stroke-width="1" />`,
        );
    }

    for (let index = 0; index <= 6; index += 1) {
        const ratio = index / 6;
        const x = margin.left + plotWidth * ratio;
        const value = timeStart + timeSpan * ratio;
        parts.push(
            `<text x="${x}" y="${height - 10}" fill="#5f7887" font-size="12" font-family="'Source Sans 3', sans-serif" text-anchor="middle">${escapeSvgText(`${value.toFixed(1)}s`)}</text>`,
        );
    }

    channels.forEach((channelName, channelIndex) => {
        const rowTop = margin.top + rowHeight * channelIndex;
        const rowCenter = rowTop + rowHeight / 2;
        const signal = centeredSignals[channelIndex] || [];
        const amplitude = signal.reduce((maxValue, value) => Math.max(maxValue, Math.abs(Number(value))), 0) || 1;
        const scaleY = (rowHeight * 0.34) / amplitude;
        const points = signal
            .map((value, sampleIndex) => {
                const x = margin.left + (plotWidth * sampleIndex) / Math.max(signal.length - 1, 1);
                const y = rowCenter - Number(value) * scaleY;
                return `${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(" ");

        parts.push(
            `<line x1="${margin.left}" y1="${rowCenter}" x2="${margin.left + plotWidth}" y2="${rowCenter}" stroke="rgba(199, 220, 227, 0.9)" stroke-width="1" />`,
        );
        parts.push(
            `<text x="12" y="${rowCenter + 4}" fill="#1c3342" font-size="13" font-family="'Source Sans 3', sans-serif">${escapeSvgText(channelName)}</text>`,
        );
        if (points) {
            parts.push(
                `<polyline points="${points}" fill="none" stroke="#163f59" stroke-width="1.25" stroke-linejoin="round" stroke-linecap="round" />`,
            );
        }
    });

    parts.push("</svg>");
    plot.style.height = `${height}px`;
    plot.innerHTML = parts.join("");
}

async function loadWaveformPreview(viewer) {
    const recordingId = viewer.getAttribute("data-recording-id");
    const startInput = viewer.querySelector("[data-waveform-start]");
    const durationInput = viewer.querySelector("[data-waveform-duration]");
    const channelsInput = viewer.querySelector("[data-waveform-channels]");
    const loadButton = viewer.querySelector("[data-waveform-load]");

    if (!recordingId || !startInput || !durationInput) {
        return;
    }

    const startSec = Number(startInput.value || 0);
    const durationSec = Number(durationInput.value || viewer.getAttribute("data-default-duration") || 30);
    const channels = channelsInput ? channelsInput.value.trim() : "";

    const url = new URL(`/api/recordings/${recordingId}/preview`, window.location.origin);
    url.searchParams.set("start_sec", Number.isFinite(startSec) ? String(startSec) : "0");
    url.searchParams.set("duration_sec", Number.isFinite(durationSec) ? String(durationSec) : "30");
    if (channels) {
        url.searchParams.set("channels", channels);
    }

    if (loadButton) {
        loadButton.disabled = true;
    }
    setWaveformStatus(viewer, "Loading EEG preview...");
    setWaveformMessages(viewer, []);

    try {
        const response = await fetch(url.toString(), {
            headers: { Accept: "application/json" },
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Unable to load the EEG waveform preview.");
        }

        startInput.max = String(Math.max(payload.total_duration_sec - payload.duration_sec, 0));
        renderWaveformPlot(viewer, payload);

        const messages = [];
        if (Array.isArray(payload.missing_channels) && payload.missing_channels.length > 0) {
            messages.push(`Unavailable channels ignored: ${payload.missing_channels.join(", ")}`);
        }
        if (Array.isArray(payload.available_channels) && payload.available_channels.length > 0) {
            messages.push(`Available EDF channels: ${payload.available_channels.join(", ")}`);
        }
        setWaveformMessages(viewer, messages);
    } catch (error) {
        const plot = viewer.querySelector("[data-waveform-plot]");
        clearWaveformCanvas(plot);
        summarizeWaveformChannels(viewer, []);
        setWaveformMessages(viewer, []);
        setWaveformStatus(
            viewer,
            error instanceof Error ? error.message : "Unable to load the EEG waveform preview.",
            true,
        );
    } finally {
        if (loadButton) {
            loadButton.disabled = false;
        }
    }
}

function initializeWaveformViewer() {
    const viewers = document.querySelectorAll("[data-waveform-viewer]");
    viewers.forEach((viewer) => {
        const loadButton = viewer.querySelector("[data-waveform-load]");
        if (loadButton) {
            loadButton.addEventListener("click", () => {
                loadWaveformPreview(viewer);
            });
        }

        loadWaveformPreview(viewer);
    });
}

function initializePrintAction() {
    const button = document.querySelector("[data-print-report]");
    if (!button) {
        return;
    }

    button.addEventListener("click", () => {
        window.print();
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initializeDeleteConfirmations();
    initializeLoadingButtons();
    initializeFileInputs();
    initializeRiskChart();
    initializeWaveformViewer();
    initializePrintAction();
});

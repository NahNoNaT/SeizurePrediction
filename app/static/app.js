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

function buildWaveformTraces(payload) {
    const centeredSignals = payload.signals.map((signal) => centerWaveformTrace(signal));
    const amplitudes = centeredSignals.map((signal) =>
        signal.reduce((maxValue, value) => Math.max(maxValue, Math.abs(value)), 0),
    );
    const spacing = Math.max(...amplitudes, 1) * 3;
    const channelCount = payload.channels.length;
    const tickvals = [];
    const traces = payload.channels.map((channelName, index) => {
        const offset = (channelCount - index - 1) * spacing;
        tickvals.push(offset);
        return {
            type: "scattergl",
            mode: "lines",
            name: channelName,
            x: payload.times,
            y: centeredSignals[index].map((value) => value + offset),
            line: {
                color: "#163f59",
                width: 1.1,
            },
            hovertemplate: `${channelName}<br>Time: %{x:.2f}s<br>Amplitude: %{customdata:.2f}<extra></extra>`,
            customdata: centeredSignals[index],
        };
    });

    return { traces, tickvals, spacing };
}

function renderWaveformPlot(viewer, payload) {
    const plot = viewer.querySelector("[data-waveform-plot]");
    if (!plot || typeof Plotly === "undefined") {
        setWaveformStatus(viewer, "Waveform plotting is unavailable in this browser.", true);
        return;
    }

    if (!Array.isArray(payload.channels) || payload.channels.length === 0 || !Array.isArray(payload.times) || payload.times.length === 0) {
        Plotly.purge(plot);
        setWaveformStatus(viewer, "No waveform samples were returned for the requested preview window.", true);
        summarizeWaveformChannels(viewer, []);
        return;
    }

    const { traces, tickvals, spacing } = buildWaveformTraces(payload);
    const layout = {
        height: Math.max(380, payload.channels.length * 92),
        margin: { l: 92, r: 28, t: 18, b: 54 },
        paper_bgcolor: "#f7fbfc",
        plot_bgcolor: "#ffffff",
        showlegend: false,
        dragmode: "pan",
        xaxis: {
            title: "Time (s)",
            gridcolor: "rgba(95, 120, 135, 0.16)",
            zeroline: false,
            linecolor: "#c7dce3",
        },
        yaxis: {
            tickmode: "array",
            tickvals,
            ticktext: payload.channels,
            showgrid: false,
            zeroline: false,
            linecolor: "#c7dce3",
            range: [-spacing * 0.75, tickvals[0] + spacing * 0.75],
        },
        font: {
            family: "'Source Sans 3', sans-serif",
            color: "#1c3342",
        },
    };
    const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    };

    Plotly.react(plot, traces, layout, config);
    setWaveformStatus(
        viewer,
        `Showing ${payload.duration_sec.toFixed(1)}s from ${payload.start_sec.toFixed(1)}s at ${payload.sampling_rate.toFixed(1)} Hz.`,
    );
    summarizeWaveformChannels(viewer, payload.channels);
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
        if (plot && typeof Plotly !== "undefined") {
            Plotly.purge(plot);
        }
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

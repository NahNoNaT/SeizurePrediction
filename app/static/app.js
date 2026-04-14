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

function downsampleArray(items, maxItems) {
    if (!Array.isArray(items) || items.length <= maxItems) {
        return items;
    }

    const step = Math.ceil(items.length / maxItems);
    return items.filter((_, index) => index % step === 0 || index === items.length - 1);
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

    points = downsampleArray(points, 600);

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
                    pointRadius(context) {
                        const size = context?.dataset?.data?.length || 0;
                        return size <= 2 ? 3 : 0;
                    },
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
                    pointRadius(context) {
                        const size = context?.dataset?.data?.length || 0;
                        return size <= 2 ? 3 : 0;
                    },
                    pointHoverRadius: 4,
                    pointBackgroundColor: "#157d84",
                    spanGaps: false,
                    borderWidth: 3,
                },
            ],
        },
        options: {
            maintainAspectRatio: false,
            animation: false,
            normalized: true,
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

    const preview = channels.slice(0, 8).join(", ");
    const suffix = channels.length > 8 ? ` +${channels.length - 8} more` : "";
    summary.textContent = `Displayed channels: ${preview}${suffix}`;
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
    const maxPointsPerChannel = Math.max(Math.floor(plotWidth / 2), 300);
    const centeredSignals = payload.signals.map((signal) => downsampleArray(centerWaveformTrace(signal), maxPointsPerChannel));

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
            const preview = payload.available_channels.slice(0, 12).join(", ");
            const suffix = payload.available_channels.length > 12 ? ` ... (+${payload.available_channels.length - 12} more)` : "";
            messages.push(`Available EDF channels: ${preview}${suffix}`);
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
        setWaveformStatus(viewer, "Click Load Preview to render a short EEG window.");
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

function formatReplayTime(seconds) {
    const totalSeconds = Math.max(Math.round(Number(seconds) || 0), 0);
    const minutes = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    if (hours > 0) {
        return `${hours}:${String(remainingMinutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }
    return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function initializeReplayMode() {
    const root = document.querySelector("[data-replay-app]");
    if (!root || typeof Chart === "undefined") {
        return;
    }

    const uploadForm = root.querySelector("[data-replay-upload-form]");
    const uploadStatus = root.querySelector("[data-replay-upload-status]");
    const fileMeta = root.querySelector("[data-replay-file-meta]");
    const startButton = root.querySelector("[data-replay-start]");
    const stopButton = root.querySelector("[data-replay-stop]");
    const windowInput = root.querySelector("[data-replay-window]");
    const hopInput = root.querySelector("[data-replay-hop]");
    const speedInput = root.querySelector("[data-replay-speed]");
    const stateLabel = document.querySelector("[data-replay-state]");
    const positionLabel = document.querySelector("[data-replay-position]");
    const progressLabel = document.querySelector("[data-replay-progress]");
    const channelLabel = document.querySelector("[data-replay-channel]");
    const riskLabel = document.querySelector("[data-replay-risk]");
    const channelCountLabel = document.querySelector("[data-replay-channel-count]");
    const messages = document.querySelector("[data-replay-messages]");
    const canvas = document.getElementById("replayTimelineChart");

    if (!uploadForm || !startButton || !stopButton || !windowInput || !hopInput || !speedInput || !canvas) {
        return;
    }

    let sessionId = "";
    let chart = null;
    let pollHandle = null;

    function setMessages(items) {
        if (!messages) {
            return;
        }
        messages.innerHTML = "";
        const resolved = Array.isArray(items) && items.length > 0
            ? items
            : ["Upload an EDF file and start replay to stream live risk updates."];
        resolved.forEach((item) => {
            const node = document.createElement("div");
            node.className = "message-item";
            node.textContent = item;
            messages.appendChild(node);
        });
    }

    function ensureChart() {
        if (chart) {
            return chart;
        }
        chart = new Chart(canvas, {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Replay seizure risk",
                        data: [],
                        fill: true,
                        tension: 0.3,
                        borderColor: "#165e86",
                        backgroundColor(context) {
                            return buildChartGradient(context, "rgba(22, 94, 134, 0.20)", "rgba(22, 94, 134, 0.02)");
                        },
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 3,
                    },
                ],
            },
            options: {
                maintainAspectRatio: false,
                animation: false,
                normalized: true,
                interaction: {
                    mode: "index",
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            usePointStyle: true,
                            boxWidth: 10,
                            color: "#1c3342",
                        },
                    },
                    tooltip: {
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
                            callback(value) {
                                return `${value}%`;
                            },
                        },
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 10,
                        },
                        grid: {
                            display: false,
                        },
                    },
                },
            },
        });
        return chart;
    }

    function renderState(payload) {
        const replayChart = ensureChart();
        const timeline = downsampleArray(payload.timeline || [], 240);
        replayChart.data.labels = timeline.map((point) => formatReplayTime(point.end_sec));
        replayChart.data.datasets[0].data = timeline.map((point) => Number(point.risk_score) * 100);
        replayChart.update("none");

        if (stateLabel) {
            stateLabel.textContent = payload.status || "idle";
        }
        if (positionLabel) {
            positionLabel.textContent = `${formatReplayTime(payload.replay_position_sec)} / ${formatReplayTime(payload.total_duration_sec)}`;
        }
        if (progressLabel) {
            progressLabel.textContent = `${payload.processed_windows} / ${payload.total_windows}`;
        }
        if (channelLabel) {
            channelLabel.textContent = payload.latest_top_channel || "-";
        }
        if (riskLabel) {
            riskLabel.textContent = payload.latest_risk_score == null ? "-" : `${(Number(payload.latest_risk_score) * 100).toFixed(1)}%`;
        }
        if (channelCountLabel) {
            channelCountLabel.textContent = String((payload.available_channels || []).length);
        }

        const info = [
            `Replay window: ${payload.window_sec}s, hop ${payload.hop_sec}s, speed ${payload.replay_speed}x.`,
        ];
        if (payload.latest_top_channel) {
            info.push(`Highest-scoring channel in the latest window: ${payload.latest_top_channel}.`);
        }
        if (payload.error) {
            info.push(payload.error);
        }
        setMessages(info);

        const isRunning = payload.status === "running";
        startButton.disabled = !sessionId || isRunning;
        stopButton.disabled = !isRunning;
    }

    async function pollState() {
        if (!sessionId) {
            return;
        }
        try {
            const response = await fetch(`/api/replay/${sessionId}`, {
                headers: { Accept: "application/json" },
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "Replay polling failed.");
            }
            renderState(payload);
            if (payload.status === "running") {
                pollHandle = window.setTimeout(pollState, 1000);
            }
        } catch (error) {
            setMessages([error instanceof Error ? error.message : "Replay polling failed."]);
            startButton.disabled = false;
            stopButton.disabled = true;
        }
    }

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        startButton.disabled = true;
        stopButton.disabled = true;
        uploadStatus.textContent = "Uploading EDF replay source...";
        fileMeta.textContent = "";
        setMessages(["Uploading EDF replay source."]);

        try {
            const response = await fetch("/api/replay/upload", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "Replay upload failed.");
            }
            sessionId = payload.session_id;
            uploadStatus.textContent = "Replay source ready.";
            fileMeta.textContent = `${payload.file_name} | ${formatReplayTime(payload.total_duration_sec)} | ${payload.available_channels.length} channels`;
            startButton.disabled = false;
            setMessages([payload.message]);
        } catch (error) {
            sessionId = "";
            const resolvedMessage = error instanceof Error ? error.message : "Replay upload failed.";
            uploadStatus.textContent = resolvedMessage;
            setMessages([resolvedMessage]);
        }
    });

    startButton.addEventListener("click", async () => {
        if (!sessionId) {
            return;
        }
        if (pollHandle) {
            window.clearTimeout(pollHandle);
            pollHandle = null;
        }
        startButton.disabled = true;
        stopButton.disabled = true;
        setMessages(["Starting EDF replay."]);

        try {
            const response = await fetch(`/api/replay/${sessionId}/start`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json",
                },
                body: JSON.stringify({
                    window_sec: Number(windowInput.value || 10),
                    hop_sec: Number(hopInput.value || 2.5),
                    replay_speed: Number(speedInput.value || 5),
                }),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "Replay could not be started.");
            }
            renderState(payload);
            if (payload.status === "running") {
                pollHandle = window.setTimeout(pollState, 1000);
            }
        } catch (error) {
            startButton.disabled = false;
            setMessages([error instanceof Error ? error.message : "Replay could not be started."]);
        }
    });

    stopButton.addEventListener("click", async () => {
        if (!sessionId) {
            return;
        }
        if (pollHandle) {
            window.clearTimeout(pollHandle);
            pollHandle = null;
        }
        try {
            const response = await fetch(`/api/replay/${sessionId}/stop`, {
                method: "POST",
                headers: { Accept: "application/json" },
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "Replay stop failed.");
            }
            renderState(payload);
        } catch (error) {
            setMessages([error instanceof Error ? error.message : "Replay stop failed."]);
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initializeDeleteConfirmations();
    initializeLoadingButtons();
    initializeFileInputs();
    initializeRiskChart();
    initializeWaveformViewer();
    initializePrintAction();
    initializeReplayMode();
});

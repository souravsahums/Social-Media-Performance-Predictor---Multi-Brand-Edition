// ── DOM refs ──
const form = document.getElementById('predict-form');
const captionEl = document.getElementById('caption');
const captionStats = document.getElementById('caption-stats');
const mediaTypeEl = document.getElementById('media_type');
const durationGroup = document.getElementById('duration-group');
const collabCheck = document.getElementById('is_collab');
const collabGroup = document.getElementById('collab-group');
const btnPredict = document.getElementById('btn-predict');
const resultsSection = document.getElementById('results-section');
const urlInputSection = document.getElementById('url-input-section');
const btnPredictUrl = document.getElementById('btn-predict-url');

// ── Input mode toggle ──
function setInputMode(mode) {
    const manualBtn = document.getElementById('mode-manual');
    const urlBtn = document.getElementById('mode-url');
    if (mode === 'url') {
        urlInputSection.style.display = '';
        form.style.display = 'none';
        manualBtn.classList.remove('active');
        urlBtn.classList.add('active');
    } else {
        urlInputSection.style.display = 'none';
        form.style.display = '';
        manualBtn.classList.add('active');
        urlBtn.classList.remove('active');
    }
}

// ── URL prediction ──
async function predictFromUrl() {
    const urlInput = document.getElementById('post_url');
    const url = urlInput.value.trim();
    if (!url) {
        alert('Please enter an Instagram post URL.');
        return;
    }

    btnPredictUrl.disabled = true;
    btnPredictUrl.textContent = '⏳ Predicting...';

    try {
        const resp = await fetch('/api/predict-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Prediction failed');
        }
        const data = await resp.json();
        renderResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btnPredictUrl.disabled = false;
        btnPredictUrl.textContent = '🔮 Predict from URL';
    }
}

// ── Toggle duration field based on media type ──
mediaTypeEl.addEventListener('change', () => {
    durationGroup.style.display = mediaTypeEl.value === 'reel' ? '' : 'none';
});

// ── Toggle collaborator field ──
collabCheck.addEventListener('change', () => {
    collabGroup.style.display = collabCheck.checked ? '' : 'none';
});

// ── Live caption stats ──
captionEl.addEventListener('input', () => {
    const text = captionEl.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const hashtags = (text.match(/#\w+/g) || []).length;
    const mentions = (text.match(/@\w+/g) || []).length;
    captionStats.textContent = `${words} words · ${hashtags} hashtags · ${mentions} mentions`;
});

// ── Form submit ──
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    btnPredict.disabled = true;
    btnPredict.textContent = '⏳ Predicting...';

    const formData = new FormData(form);
    // FormData doesn't include unchecked checkboxes
    if (!collabCheck.checked) {
        formData.set('is_collab', 'false');
    } else {
        formData.set('is_collab', 'true');
    }

    try {
        const resp = await fetch('/api/predict-simple', {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Prediction failed');
        }
        const data = await resp.json();
        renderResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btnPredict.disabled = false;
        btnPredict.textContent = '🔮 Predict Performance';
    }
});

// ── Render results ──
let lastPredictionData = null;

function renderResults(data) {
    lastPredictionData = data;
    resultsSection.style.display = '';

    // Reset feedback state
    document.getElementById('correction-form').style.display = 'none';
    document.getElementById('feedback-msg').style.display = 'none';
    document.getElementById('btn-correct').disabled = false;
    document.getElementById('btn-wrong').disabled = false;

    // Source info (URL predictions)
    let sourceEl = document.getElementById('source-info');
    if (!sourceEl) {
        sourceEl = document.createElement('div');
        sourceEl.id = 'source-info';
        resultsSection.insertBefore(sourceEl, resultsSection.firstChild);
    }
    if (data.source_url) {
        sourceEl.style.display = '';
        sourceEl.className = 'source-info';
        sourceEl.innerHTML = `<strong>📌 Matched Post:</strong> <a href="${escapeHtml(data.source_url)}" target="_blank">${escapeHtml(data.source_brand)}</a>` +
            (data.source_caption_preview ? ` — <em>${escapeHtml(data.source_caption_preview.substring(0, 80))}${data.source_caption_preview.length > 80 ? '…' : ''}</em>` : '');
    } else {
        sourceEl.style.display = 'none';
    }

    // Prediction badge
    const badge = document.getElementById('prediction-badge');
    badge.className = 'prediction-badge ' + data.prediction;
    document.getElementById('pred-label').textContent = data.prediction;
    document.getElementById('pred-conf').textContent =
        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    // Probability bars
    const probs = data.probabilities;
    setBar('prob-low', 'prob-low-val', probs.low);
    setBar('prob-medium', 'prob-med-val', probs.medium);
    setBar('prob-high', 'prob-high-val', probs.high);

    // Explanations
    const explList = document.getElementById('explanation-list');
    explList.innerHTML = '';
    (data.explanation || []).forEach((item) => {
        const icon = item.impact === 'positive' ? '✅'
            : item.impact === 'negative' ? '⚠️' : 'ℹ️';
        const div = document.createElement('div');
        div.className = 'explanation-item ' + item.impact;
        div.innerHTML = `
            <span class="expl-icon">${icon}</span>
            <div class="expl-body">
                <div class="expl-factor">${escapeHtml(item.factor)}</div>
                <div class="expl-detail">${escapeHtml(item.detail)}</div>
            </div>`;
        explList.appendChild(div);
    });

    // Brand context
    const ctx = data.brand_context || {};
    const ctxEl = document.getElementById('brand-context');
    ctxEl.innerHTML = '';
    const statsToShow = [
        ['mean', 'Avg ER%'],
        ['median', 'Median ER%'],
        ['p25', '25th Pct'],
        ['p75', '75th Pct'],
        ['count', '# Posts'],
    ];
    statsToShow.forEach(([key, label]) => {
        if (ctx[key] !== undefined) {
            const val = key === 'count' ? ctx[key] : ctx[key].toFixed(2) + '%';
            const div = document.createElement('div');
            div.className = 'brand-stat';
            div.innerHTML = `<div class="stat-val">${val}</div>
                             <div class="stat-name">${label}</div>`;
            ctxEl.appendChild(div);
        }
    });
    if (Object.keys(ctx).length === 0) {
        ctxEl.innerHTML = '<p style="color:var(--text-dim)">No historical data for this brand. Using global baseline.</p>';
    }

    // SLM Reasoning section
    const slmSection = document.getElementById('slm-section');
    if (data.slm_prediction) {
        slmSection.style.display = '';
        const slmBadge = document.getElementById('slm-prediction');
        slmBadge.textContent = `SLM says: ${data.slm_prediction.toUpperCase()} (${(data.slm_confidence * 100).toFixed(0)}%)`;
        slmBadge.className = 'slm-badge ' + data.slm_prediction;

        const slmReasoning = document.getElementById('slm-reasoning');
        slmReasoning.innerHTML = '';
        (data.slm_reasoning || []).forEach((item) => {
            const div = document.createElement('div');
            div.className = 'explanation-item ' + (item.points > 0 ? 'positive' : item.points < 0 ? 'negative' : 'neutral');
            div.innerHTML = `
                <span class="expl-icon">${item.points > 0 ? '✅' : item.points < 0 ? '⚠️' : 'ℹ️'}</span>
                <div class="expl-body">
                    <div class="expl-factor">${escapeHtml(item.factor)} (${item.points > 0 ? '+' : ''}${item.points}pts)</div>
                    <div class="expl-detail">${escapeHtml(item.explanation)}</div>
                </div>`;
            slmReasoning.appendChild(div);
        });
    } else {
        slmSection.style.display = 'none';
    }

    // Drift warning
    const driftEl = document.getElementById('drift-warning');
    if (data.drift_warning && data.drift_warning !== 'low') {
        driftEl.style.display = '';
        driftEl.className = 'drift-warning ' + data.drift_warning;
        driftEl.textContent = data.drift_warning === 'high'
            ? '⚠️ High data drift detected — prediction may be unreliable.'
            : '⚡ Moderate data anomaly detected — take prediction with caution.';
    } else {
        driftEl.style.display = 'none';
    }

    // Scroll to results on mobile
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setBar(barId, valId, prob) {
    document.getElementById(barId).style.width = (prob * 100) + '%';
    document.getElementById(valId).textContent = (prob * 100).toFixed(1) + '%';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// ── Feedback functions ──
function showCorrectionForm() {
    document.getElementById('correction-form').style.display = '';
}

async function submitFeedback(isCorrect) {
    if (!lastPredictionData) return;

    const msgEl = document.getElementById('feedback-msg');

    if (isCorrect) {
        msgEl.textContent = '✅ Thanks for confirming!';
        msgEl.style.display = '';
        document.getElementById('btn-correct').disabled = true;
        document.getElementById('btn-wrong').disabled = true;
        return;
    }

    const correctLabel = document.getElementById('correct-label').value;
    const body = {
        prediction_id: '',
        predicted_label: lastPredictionData.prediction,
        correct_label: correctLabel,
        brand: document.getElementById('brand').value,
        caption: document.getElementById('caption').value,
        media_type: document.getElementById('media_type').value,
        features: lastPredictionData.features_used || {},
    };

    try {
        const resp = await fetch('/api/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        const result = await resp.json();
        msgEl.textContent = result.message || 'Feedback submitted.';
        msgEl.style.display = '';
        document.getElementById('btn-correct').disabled = true;
        document.getElementById('btn-wrong').disabled = true;
        document.getElementById('correction-form').style.display = 'none';
    } catch (err) {
        msgEl.textContent = 'Error submitting feedback: ' + err.message;
        msgEl.style.display = '';
    }
}

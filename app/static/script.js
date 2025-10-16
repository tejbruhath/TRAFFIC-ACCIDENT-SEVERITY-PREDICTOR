async function predict() {
  const get = id => document.getElementById(id).value;
  const num = v => (v === '' || isNaN(Number(v)) ? null : Number(v));

  const payload = {
    state_ut_city: get('state_ut_city'),
    road_accidents_cases: num(get('road_accidents_cases')),
    road_accidents_injured: num(get('road_accidents_injured')),
    road_accidents_died: num(get('road_accidents_died')),
    total_traffic_accidents_cases: num(get('total_traffic_accidents_cases')),
    total_traffic_accidents_injured: num(get('total_traffic_accidents_injured')),
    total_traffic_accidents_died: num(get('total_traffic_accidents_died')),
  };

  if (document.getElementById('compute_ratios').checked) {
    const casesT = payload.total_traffic_accidents_cases;
    const diedT = payload.total_traffic_accidents_died;
    const casesR = payload.road_accidents_cases;
    const diedR = payload.road_accidents_died;
    if (casesT && diedT) payload.fatality_ratio_total = diedT / casesT;
    if (casesR && diedR) payload.fatality_ratio_road = diedR / casesR;
  }

  const resDiv = document.getElementById('result');
  resDiv.style.display = 'block';
  resDiv.textContent = 'Predicting...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediction failed');
    resDiv.innerHTML = `<b>Prediction:</b> ${data.prediction} ` + (data.confidence != null ? `(confidence ${data.confidence.toFixed(3)})` : '');
  } catch (err) {
    resDiv.textContent = `Error: ${err.message}`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('predictBtn').addEventListener('click', predict);
});

const API_URL = 'http://localhost:8000';
let regChart = null;
let clsChart = null;

function switchTab(type) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.dashboard').forEach(dash => dash.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(`${type}-dash`).classList.add('active');
}

async function runRegression() {
    const features = JSON.parse(document.getElementById('reg-features').value);
    const targets = JSON.parse(document.getElementById('reg-targets').value);

    try {
        const response = await fetch(`${API_URL}/regression`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features, targets })
        });

        const data = await response.json();
        displayRegressionResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Backend not responding. Please make sure main.py is running.');
    }
}

function displayRegressionResults(data) {
    document.getElementById('reg-results').style.display = 'block';
    document.getElementById('reg-mse').innerText = data.mse.toFixed(4);
    document.getElementById('reg-mae').innerText = data.mae.toFixed(4);
    document.getElementById('reg-r2').innerText = data.r2.toFixed(4);

    if (regChart) regChart.destroy();
    
    const ctx = document.getElementById('regChart').getContext('2d');
    regChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.predictions.map((_, i) => i + 1),
            datasets: [{
                label: 'Predictions',
                data: data.predictions,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: 'white' } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: 'white' } },
                x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: 'white' } }
            }
        }
    });
}

async function runClassification() {
    const features = JSON.parse(document.getElementById('cls-features').value);
    const targets = JSON.parse(document.getElementById('cls-targets').value);

    try {
        const response = await fetch(`${API_URL}/classification`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features, targets })
        });

        const data = await response.json();
        displayClassificationResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Backend not responding.');
    }
}

function displayClassificationResults(data) {
    document.getElementById('cls-results').style.display = 'block';
    document.getElementById('cls-acc').innerText = (data.accuracy * 100).toFixed(1) + '%';
    document.getElementById('cls-f1').innerText = data.f1_score.toFixed(4);
    document.getElementById('cls-prec').innerText = data.precision.toFixed(4);

    if (clsChart) clsChart.destroy();

    const ctx = document.getElementById('clsChart').getContext('2d');
    
    // Simple bar chart for predictions
    const counts = data.predictions.reduce((acc, val) => {
        acc[val] = (acc[val] || 0) + 1;
        return acc;
    }, {});

    clsChart = new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: Object.keys(counts).map(k => `Class ${k}`),
            datasets: [{
                data: Object.values(counts),
                backgroundColor: ['#f43f5e', '#a855f7', '#6366f1', '#10b981']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: 'white' } }
            },
            scales: {
                r: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { backdropColor: 'transparent', color: 'white' } }
            }
        }
    });
}

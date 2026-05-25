const form = document.getElementById('predictionForm');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const loadingDiv = document.getElementById('loading');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Hide previous results
    resultDiv.classList.remove('show');
    errorDiv.classList.remove('show');
    loadingDiv.classList.add('show');

    const formData = {
        Pregnancies: parseInt(document.getElementById('pregnancies').value),
        Glucose: parseInt(document.getElementById('glucose').value),
        BloodPressure: parseInt(document.getElementById('bloodPressure').value),
        SkinThickness: parseInt(document.getElementById('skinThickness').value),
        Insulin: parseInt(document.getElementById('insulin').value),
        BMI: parseFloat(document.getElementById('bmi').value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById('dpf').value),
        Age: parseInt(document.getElementById('age').value)
    };

    try {
        const response = await fetch('https://diabetes-prediction-bl6a.onrender.com/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(formData)
});

        if (!response.ok) {
            throw new Error('Prediction failed. Please check your input values.');
        }

        const data = await response.json();

        // Display results
        const predictionLabel = data.class_label;
        const probability = (data.probability * 100).toFixed(2);

        document.getElementById('predictionLabel').textContent = predictionLabel;
        document.getElementById('predictionLabel').className =
            predictionLabel === 'Diabetic' ? 'prediction-positive' : 'prediction-negative';
        document.getElementById('predictionValue').textContent = data.prediction;
        document.getElementById('probabilityValue').textContent = probability + '%';

        resultDiv.classList.add('show');
    } catch (error) {
        errorDiv.textContent = '❌ ' + error.message;
        errorDiv.classList.add('show');
    } finally {
        loadingDiv.classList.remove('show');
    }
});

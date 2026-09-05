document.addEventListener('DOMContentLoaded', function () {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    // Click on upload area triggers file dialog
    uploadArea.addEventListener('click', function () {
        fileInput.click();
    });

    // Drag & drop events
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.style.background = '#e2edf8';
        uploadArea.style.borderColor = '#1a73e8';
    });

    uploadArea.addEventListener('dragleave', function () {
        uploadArea.style.background = '#fafdff';
        uploadArea.style.borderColor = '#b8d0e0';
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.style.background = '#fafdff';
        uploadArea.style.borderColor = '#b8d0e0';

        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // ── Main file handler ──
    function handleFile(file) {
        // Basic validation
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            alert('❌ Please upload a valid image (JPG, PNG, JPEG).');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('❌ File is too large. Please upload an image under 10 MB.');
            return;
        }

        // Show loading
        loading.style.display = 'block';
        results.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        // Get patient name from input field
        const patientNameInput = document.getElementById('patientName');
        const patientName = patientNameInput.value.trim() || 'Anonymous';
        formData.append('patient_name', patientName);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';

            if (!data.success) {
                // Show quality check failure or other error
                alert('⚠️ ' + (data.quality_check || data.error || 'Analysis failed. Please try again.'));
                return;
            }

            // ── Display results ──
            // Original image
            document.getElementById('originalImage').src = URL.createObjectURL(file);

            // Grad-CAM image (if available)
            const gradcamImg = document.getElementById('gradcamImage');
            if (data.gradcam_image) {
                gradcamImg.src = data.gradcam_image;
            } else {
                gradcamImg.src = '';
                gradcamImg.alt = 'Grad-CAM not available';
            }

            // Prediction
            document.getElementById('className').textContent = data.prediction;

            // Confidence
            const confPercent = (data.confidence * 100).toFixed(0);
            document.getElementById('confidenceValue').textContent = confPercent + '%';
            document.getElementById('confidenceFill').style.width = confPercent + '%';

            // Risk badge
            const riskBadge = document.getElementById('riskBadge');
            const isHigh = data.risk_level.includes('High');
            riskBadge.textContent = (isHigh ? '🔴 ' : '🟢 ') + data.risk_level;
            riskBadge.className = 'badge ' + (isHigh ? 'badge-high' : 'badge-low');

            // Recommendation
            document.getElementById('recommendationText').textContent = data.recommendation;

            // Trend Alert
            const trendAlert = document.getElementById('trendAlert');
            const trendMessage = document.getElementById('trendMessage');
            if (data.trend_message) {
                trendAlert.style.display = 'block';
                trendMessage.textContent = data.trend_message;
            } else {
                trendAlert.style.display = 'none';
            }

            // Show results
            results.style.display = 'block';

            // Scroll to results
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        })
        .catch(err => {
            loading.style.display = 'none';
            alert('❌ Server error. Please make sure Flask is running.\n' + err.message);
            console.error(err);
        });
    }
});
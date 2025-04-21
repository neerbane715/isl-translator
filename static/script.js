// DOM Elements
const webcam = document.getElementById('webcam');
const cameraPopup = document.getElementById('cameraPopup');
const closePopup = document.getElementById('closePopup');
const retryBtn = document.getElementById('retryCamera');
const gestureOutput = document.getElementById('gestureOutput');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const placeholderText = document.getElementById('placeholderText');
const speakBtn = document.getElementById('speakBtn');
const languageBtns = document.querySelectorAll('.language-btn');

// Speech synthesis setup
const speechSynthesis = window.speechSynthesis;
let currentLanguage = 'en-US'; // Default language

// Enhanced popup handlers
function setupPopupHandlers() {
    try {
        const permPopup = document.getElementById('permissionPopup');
        const closePermPopup = document.getElementById('closePermPopup');
        const openSettingsBtn = document.getElementById('openSettings');

        if (permPopup && closePermPopup && openSettingsBtn) {
            closePermPopup.addEventListener('click', () => {
                permPopup.classList.add('hidden');
            });

            openSettingsBtn.addEventListener('click', () => {
                try {
                    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
                    const message = isMobile 
                        ? 'Go to Settings > Site Settings > Camera to enable permissions'
                        : 'Click the camera icon in your address bar to manage permissions';
                    alert(`Please enable camera access:\n${message}`);
                } catch (e) {
                    console.error('Error showing settings help:', e);
                    alert('Please check your browser settings to enable camera permissions.');
                }
            });
        }

        if (cameraPopup && closePopup && retryBtn) {
            closePopup.addEventListener('click', () => {
                cameraPopup.classList.add('hidden');
            });

            retryBtn.addEventListener('click', async () => {
                cameraPopup.classList.add('hidden');
                await setupCamera();
            });
        }
    } catch (error) {
        console.error('Error setting up popup handlers:', error);
    }
}

const loadingOverlay = document.getElementById('loadingOverlay');
const flipCamera = document.getElementById('flipCamera');
const startCameraBtn = document.getElementById('startCamera');
const stopCameraBtn = document.getElementById('stopCamera');

// Camera state
let currentStream = null;
let isBackCamera = false;
let predictionInterval = null;

// Prediction state
let lastPrediction = '_';
let predictionCount = 0;
const PREDICTION_THRESHOLD = 3;
let detectedGestures = []; // Array to store recent gestures
let detectedGesturesHindi = []; // Array to store Hindi translations
const MAX_GESTURES = 5;

// Function to speak text
function speakText(text) {
    // Stop any ongoing speech
    speechSynthesis.cancel();

    if (text && text.trim() !== '') {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = currentLanguage;
        utterance.rate = 0.9; // Slightly slower than normal
        utterance.pitch = 1;
        
        // Get available voices
        const voices = speechSynthesis.getVoices();
        
        // For Hindi, specifically look for a Hindi voice
        if (currentLanguage === 'hi-IN') {
            const hindiVoice = voices.find(voice => 
                voice.lang.startsWith('hi') || 
                voice.name.toLowerCase().includes('hindi')
            );
            if (hindiVoice) {
                utterance.voice = hindiVoice;
            }
        } else {
            // For English, find an appropriate English voice
            const englishVoice = voices.find(voice => 
                voice.lang.startsWith('en')
            );
            if (englishVoice) {
                utterance.voice = englishVoice;
            }
        }

        speechSynthesis.speak(utterance);
    }
}

// Language selection handler
function setupLanguageButtons() {
    languageBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            languageBtns.forEach(b => b.classList.remove('bg-indigo-800'));
            // Add active class to clicked button
            btn.classList.add('bg-indigo-800');
            
            // Set language based on button
            const language = btn.dataset.lang.toLowerCase();
            if (language === 'english') {
                currentLanguage = 'en-US';
            } else if (language === 'hindi') {
                currentLanguage = 'hi-IN';
            }
            // Update display with current language
            updateGestureDisplay();
        });
    });
}

// Speak button handler
function setupSpeakButton() {
    if (speakBtn) {
        speakBtn.addEventListener('click', () => {
            // Use Hindi translations if Hindi is selected
            const textToSpeak = currentLanguage === 'hi-IN' ? 
                detectedGesturesHindi.join(' ') : 
                detectedGestures.join(' ');
                
            if (textToSpeak) {
                speakText(textToSpeak);
            }
        });
    }
}

// Camera setup
async function setupCamera() {
    const permPopup = document.getElementById('permissionPopup');
    const cameraPopup = document.getElementById('cameraPopup');

    try {
        showLoading();

        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }

        currentStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: isBackCamera ? 'environment' : 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });

        webcam.srcObject = currentStream;

        await new Promise((resolve) => {
            webcam.onloadedmetadata = resolve;
        });

        hideLoading();
        return true;
    } catch (error) {
        console.error('Camera error:', error);
        hideLoading();

        if (error.name === 'NotAllowedError') {
            if (permPopup) {
                permPopup.classList.remove('hidden');
            } else {
                alert('Camera permission denied. Please enable camera access in your browser settings.');
            }
        } else if (error.name === 'NotFoundError' || error.name === 'OverconstrainedError') {
            if (cameraPopup) {
                cameraPopup.classList.remove('hidden');
            } else {
                alert('No compatible camera found. Please check your camera connection.');
            }
        } else {
            const errorMsg = `Camera Error: ${error.message}`;
            const errorToast = document.getElementById('errorToast');
            if (errorToast) {
                errorToast.textContent = errorMsg;
                errorToast.classList.remove('hidden');
                setTimeout(() => {
                    errorToast.classList.add('hidden');
                }, 5000);
            } else {
                alert(errorMsg);
            }
        }
        return false;
    }
}

// Function to capture frame and send to server
async function captureAndPredict() {
    if (!webcam.videoWidth || !currentStream || !currentStream.active) {
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext('2d');
    
    try {
        if (!isBackCamera) {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(webcam, 0, 0);

        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        const response = await fetch('https://isl-translator-2wii.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',  
            },
            credentials: "include",
            body: JSON.stringify({
                image: imageData
            })
        });

        const result = await response.json();
        
        if (result.error) {
            console.error('Prediction error:', result.error);
            return;
        }

        if (currentStream && currentStream.active) {
            updatePredictionUI(result.prediction, result.confidence, result.hindi);
        }

    } catch (error) {
        console.error('Error sending frame to server:', error);
    }
}

// Function to update UI with prediction
function updatePredictionUI(prediction, confidence, hindiTranslation) {
    // Update confidence bar and text
    confidenceBar.style.width = `${confidence * 100}%`;
    confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

    // Only update the prediction display if confidence is above threshold
    if (confidence > 0.7) {
        if (prediction === lastPrediction) {
            predictionCount++;
            if (predictionCount >= PREDICTION_THRESHOLD && prediction !== '_') {
                // Only add to gestures if it's a new gesture or first gesture
                if (detectedGestures.length === 0 || detectedGestures[detectedGestures.length - 1] !== prediction) {
                    detectedGestures.push(prediction);
                    detectedGesturesHindi.push(hindiTranslation);
                    if (detectedGestures.length > MAX_GESTURES) {
                        detectedGestures.shift(); // Remove oldest gesture
                        detectedGesturesHindi.shift();
                    }
                    updateGestureDisplay();
                }
            }
        } else {
            // Reset counter for new gesture
            predictionCount = 1;
            lastPrediction = prediction;
        }
    } else {
        // Reset counter if confidence is low
        predictionCount = 0;
    }
}

// Function to update the gesture display
function updateGestureDisplay() {
    if (detectedGestures.length > 0) {
        // Create a sentence from gestures based on selected language
        const displayText = currentLanguage === 'hi-IN' ? 
            detectedGesturesHindi.join(' ') : 
            detectedGestures.join(' ');
            
        placeholderText.textContent = displayText;
        placeholderText.classList.remove('text-gray-400');
        placeholderText.classList.add('text-indigo-600');

        // Update the emoji/gesture icon
        if (gestureOutput) {
            const lastGesture = currentLanguage === 'hi-IN' ? 
                detectedGesturesHindi[detectedGesturesHindi.length - 1] : 
                detectedGestures[detectedGestures.length - 1];
                
            gestureOutput.querySelector('.text-5xl').textContent = '🤟';
            gestureOutput.querySelector('.text-sm').textContent = `Last detected: ${lastGesture}`;
        }
    } else {
        placeholderText.textContent = currentLanguage === 'hi-IN' ? 
            'आपका अनुवाद यहां दिखाई देगा' : 
            'Your translations will appear here';
        placeholderText.classList.add('text-gray-400');
        placeholderText.classList.remove('text-indigo-600');
        
        if (gestureOutput) {
            gestureOutput.querySelector('.text-5xl').textContent = '👋';
            gestureOutput.querySelector('.text-sm').textContent = 'Detected Gesture';
        }
    }
}

// Camera control functions
async function startCamera() {
    console.log('Start camera function called');
    const cameraPopup = document.getElementById('cameraPopup');
    if (cameraPopup && !cameraPopup.classList.contains('hidden')) {
        return;
    }

    try {
        if (await setupCamera()) {
            console.log('Camera setup successful');
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            
            // Reset gesture state
            detectedGestures = [];
            detectedGesturesHindi = [];
            lastPrediction = '_';
            predictionCount = 0;
            updateGestureDisplay();
            
            // Notify server to start predictions
            console.log('Notifying server to start predictions');
            try {
                const response = await fetch('https://isl-translator-2wii.onrender.com/start_predictions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                const result = await response.json();
                console.log('Server start response:', result);
                
                // Only start predictions if server acknowledges
                if (result.status === 'success') {
                    console.log('Starting prediction interval');
                    if (predictionInterval) {
                        clearInterval(predictionInterval);
                    }
                    predictionInterval = setInterval(captureAndPredict, 100); // Predict every 100ms
                }
            } catch (error) {
                console.error('Error notifying server to start predictions:', error);
            }
        }
    } catch (error) {
        console.error('Error starting camera:', error);
    }
}

async function stopCamera() {
    console.log('Stop camera function called');
    
    // Immediately stop predictions on client side
    if (predictionInterval) {
        console.log('Clearing prediction interval');
        clearInterval(predictionInterval);
        predictionInterval = null;
    }

    // Immediately stop and clear video element
    if (webcam) {
        console.log('Stopping video element');
        webcam.pause();
        webcam.srcObject = null;
    }

    // Stop the camera stream
    if (currentStream) {
        console.log('Stopping camera stream tracks');
        try {
            const tracks = currentStream.getTracks();
            tracks.forEach(track => {
                console.log('Stopping track:', track.kind);
                track.stop();
                track.enabled = false;
            });
        } catch (error) {
            console.error('Error stopping tracks:', error);
        }
        currentStream = null;
    }

    // Notify server to stop camera and predictions
    try {
        console.log('Notifying server to stop camera and predictions');
        const response = await fetch('https://isl-translator-2wii.onrender.com/stop_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        const result = await response.json();
        console.log('Server stop response:', result);
        
        if (result.status !== 'success') {
            console.error('Server failed to stop properly:', result);
        }
    } catch (error) {
        console.error('Error notifying server to stop:', error);
    }
    
    // Update UI elements
    console.log('Updating UI elements');
    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;
    
    // Reset all state
    detectedGestures = [];
    detectedGesturesHindi = [];
    lastPrediction = '_';
    predictionCount = 0;
    updateGestureDisplay();
    confidenceBar.style.width = '0%';
    confidenceText.textContent = 'Confidence: 0%';
    
    // Force a final UI refresh
    if (webcam) {
        webcam.style.display = 'none';
        setTimeout(() => {
            webcam.style.display = 'block';
            console.log('Webcam element reset complete');
        }, 100);
    }
    
    console.log('Stop camera function completed');
}

async function flipCameraHandler() {
    isBackCamera = !isBackCamera;
    if (!stopCameraBtn.disabled) {
        await startCamera();
    }
}

// UI Helpers
function showLoading() {
    if (loadingOverlay) loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    if (loadingOverlay) loadingOverlay.classList.add('hidden');
}

// Initialize event listeners
function setupEventListeners() {
    if (flipCamera) flipCamera.addEventListener('click', flipCameraHandler);
    if (startCameraBtn) startCameraBtn.addEventListener('click', startCamera);
    if (stopCameraBtn) stopCameraBtn.addEventListener('click', stopCamera);
    setupPopupHandlers();
}

// Initialize app
async function init() {
    console.log("Initializing application...");
    setupEventListeners();
    setupLanguageButtons();
    setupSpeakButton();
    stopCameraBtn.disabled = true;

    // Load voices if they haven't loaded yet
    if (speechSynthesis.getVoices().length === 0) {
        await new Promise(resolve => {
            speechSynthesis.addEventListener('voiceschanged', resolve, { once: true });
        });
    }
}

// Start when ready
if (document.readyState === 'complete') {
    init();
} else {
    document.addEventListener('DOMContentLoaded', init);
}

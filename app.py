import streamlit as st
def check_dependencies():
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        import mediapipe as mp
        from flask import Flask
        from flask_cors import CORS
        print("All required libraries are installed:")
        print(f"✓ TensorFlow version: {tf.__version__}")
        print(f"✓ OpenCV version: {cv2.__version__}")
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Mediapipe version: {mp.__version__}")
        print(f"✓ Flask installed")
        print(f"✓ Flask-CORS installed")
        return True
    except ImportError as e:
        print(f"Error: Missing required library - {str(e)}")
        print("Please install the missing library using pip install")
        return False

# Check dependencies first
if not check_dependencies():
    print("Please install all required dependencies before running the server.")
    exit(1)

from flask import Flask, request, jsonify, Response, render_template
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask_cors import CORS
import base64
import os
import signal
from threading import Event, Lock
import re
from Detection_model import mediapipe_detection, extract_keypoints, actions, model

app = Flask(__name__)

# Set correct frontend origin
# frontend_origin = 'https://isl-s6wi.onrender.com'

# CORS configuration
# CORS(app, origins=[frontend_origin], supports_credentials=True)
CORS(app, resources={r"/*": {"origins": "https://isl-s6wi.onrender.com"}}, supports_credentials=True)
# Global shutdown event
shutdown_event = Event()

# Thread safety locks
model_lock = Lock()
sequence_lock = Lock()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/interpreter')
def interpreter():
    return render_template('interpreter.html')

# Initialize sequence buffer globally
sequence = []  # Keep last 30 frames

# Initialize MediaPipe Holistic model globally
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

translations = {
    'hello': 'नमस्ते',
    'namaste': 'नमस्ते',
    'bye': 'अलविदा',
    'india': 'भारत',
    'thanks': 'धन्यवाद',
    'sorry': 'माफ़ कीजिये',
    'good': 'अच्छा',
    'yes': 'हाँ',
    'no': 'नहीं',
    '_': '_'
}

is_predicting = False

def base64_to_image(base64_string):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', base64_string)
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        print(f"Error in base64_to_image: {str(e)}")
        raise

@app.route('/start_predictions', methods=['POST'])
def start_predictions():
    global is_predicting, sequence
    print("Start predictions endpoint called")
    try:
        with sequence_lock:
            sequence = []
            is_predicting = True
        print("Prediction state initialized")
        return jsonify({
            'message': 'Predictions started successfully',
            'status': 'success'
        })
    except Exception as e:
        error_msg = f"Error starting predictions: {str(e)}"
        print(error_msg)
        return jsonify({
            'error': error_msg,
            'status': 'error'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    global sequence, is_predicting

    if not is_predicting:
        return jsonify({
            'prediction': '_',
            'hindi': '_',
            'confidence': 0.0,
            'status': 'stopped'
        })

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        try:
            frame = base64_to_image(image_data)
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

        try:
            with model_lock:
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                with sequence_lock:
                    if not is_predicting:
                        return jsonify({
                            'prediction': '_',
                            'hindi': '_',
                            'confidence': 0.0,
                            'status': 'stopped'
                        })

                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        model_input = np.expand_dims(np.array(sequence), axis=0)
                        prediction = model.predict(model_input, verbose=0)

                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = float(prediction[0][predicted_class_idx])
                        predicted_action = actions[predicted_class_idx]
                        hindi_translation = translations[predicted_action]

                        return jsonify({
                            'prediction': predicted_action,
                            'hindi': hindi_translation,
                            'confidence': confidence,
                            'status': 'active'
                        })

                    return jsonify({
                        'prediction': '_',
                        'hindi': '_',
                        'confidence': 0.0,
                        'status': 'buffering'
                    })
        except Exception as e:
            print(f"Error in prediction pipeline: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global sequence, is_predicting
    print("Stop camera endpoint called")
    try:
        is_predicting = False
        print("Predictions disabled")

        print("Acquiring sequence lock...")
        with sequence_lock:
            print("Clearing sequence buffer")
            sequence = []
        print("Sequence buffer cleared")

        try:
            print("Resetting MediaPipe Holistic model")
            holistic.reset()
            print("MediaPipe model reset successful")
        except Exception as e:
            print(f"Warning: Could not reset MediaPipe model: {str(e)}")

        with model_lock:
            try:
                model.reset_states()
                print("Model states reset")
            except Exception as e:
                print(f"Warning: Could not reset model states: {str(e)}")

        print("Stop camera operation completed successfully")
        return jsonify({
            'message': 'Camera and predictions stopped successfully',
            'status': 'success',
            'sequence_cleared': True,
            'model_reset': True,
            'predictions_disabled': True
        })
    except Exception as e:
        error_msg = f"Error stopping camera and predictions: {str(e)}"
        print(error_msg)
        return jsonify({
            'error': error_msg,
            'status': 'error'
        }), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    try:
        shutdown_event.set()
        try:
            holistic.close()
        except:
            pass

        response = jsonify({'message': 'Server shutdown initiated'})

        def shutdown_server():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                os._exit(0)
            func()

        from threading import Timer
        Timer(1.0, shutdown_server).start()

        return response
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global sequence
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        try:
            while not shutdown_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                prediction = '_'
                confidence = 0.0

                if len(sequence) == 30:
                    model_input = np.expand_dims(np.array(sequence), axis=0)
                    res = model.predict(model_input)
                    predicted_class_idx = np.argmax(res[0])
                    confidence = float(res[0][predicted_class_idx])
                    prediction = actions[predicted_class_idx]

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, prediction, (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    print("Error: Could not encode frame")
                    break

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Video feed error: {str(e)}")
        finally:
            cap.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Server starting... Press Ctrl+C to stop manually")
    try:
        app.config['PROPAGATE_EXCEPTIONS'] = True
        app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        shutdown_event.set()
    finally:
        try:
            holistic.close()
        except:
            pass




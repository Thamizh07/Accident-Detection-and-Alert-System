from flask import Flask, render_template, request
import subprocess  

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index.html')  
def index():
    return render_template('index.html')

@app.route('/run_drowsiness_detection', methods=['POST'])
def run_drowsiness_detection():
    try:
        file_path0 = r"C:\Users\thamizh\Desktop\sem 5\Six models\Drowsiness detection\dd2.py"
        subprocess.run(["python", file_path0]) 
        return 'Drowsiness detection executed successfully!', 200
    except Exception as e:
        return f'Error executing drowsiness detection: {e}', 500

@app.route('/run_lane_change_monitoring', methods=['POST'])
def run_lane_change_monitoring():
    try:
        file_path1 = r"C:\Users\thamizh\Desktop\sem 5\Six models\lane-detection-with-steer-and-departure-master\lane-detection-with-steer-and-departure-master\laneDetection.py"
        subprocess.run(["python", file_path1])
        return 'Lane change monitoring executed successfully!', 200
    except Exception as e:
        return f'Error executing lane change monitoring: {e}', 500
    
@app.route('/run_pothole_detection', methods=['POST'])
def run_pothole_detection():
    try:
        file_path2 = r"C:\Users\thamizh\Desktop\sem 5\Six models\Pothole-Detection-System-using-YOLO-Tiny-v4-main\Pothole-Detection-System-using-YOLO-Tiny-v4-main\main.py"
        subprocess.run(["python", file_path2])
        return 'Pothole detection executed successfully!', 200
    except Exception as e:
        return f'Error executing pothole detection: {e}', 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import threading
import json

app = Flask(__name__)
data = []
LINK_MQTT = "6f01648d2a274f6d979d38c4609a3d8c.s1.eu.hivemq.cloud"
PORT_MQTT = 8883
TOPIC_SUHU = "data/suhu"
TOPIC_KELEMBAPAN = "data/kelembapan"
USERNAME_MQTT = "savaqua"
PASSWORD_MQTT = "Savaqua123"
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe([(TOPIC_SUHU, 0), (TOPIC_KELEMBAPAN, 0)])

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    print(f"Received message '{payload}' on topic '{topic}'")

    if topic == TOPIC_SUHU:
        update_data('suhu', str(payload))
    elif topic == TOPIC_KELEMBAPAN:
        update_data('kelembapan', str(payload))

def update_data(key, value):
    if not data:
        new_id = 1
        new_data = {'id': new_id, 'suhu': None, 'kelembapan': None}
        data.append(new_data)
    
    latest_data = data[-1]
    latest_data[key] = value

    if latest_data['suhu'] is not None and latest_data['kelembapan'] is not None:
        new_id = latest_data['id'] + 1
        new_data = {'id': new_id, 'suhu': None, 'kelembapan': None}
        data.append(new_data)

mqtt_client.username_pw_set(USERNAME_MQTT, PASSWORD_MQTT)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

def start_mqtt():
    mqtt_client.connect(LINK_MQTT, PORT_MQTT, 60)
    mqtt_client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt)
mqtt_thread.start()

@app.route('/sensor/data', methods=['GET'])
def get_data():
    return jsonify(data)

@app.route('/sensor/data/<int:id>', methods=['GET'])
def get_data_by_id(id):
    result = next((item for item in data if item["id"] == id), None)
    if result:
        return jsonify(result)
    else:
        return jsonify({'pesan': 'Data tidak ditemukan'}), 404

@app.route('/sensor/data', methods=['POST'])
def add_data():
    req_data = request.get_json()
    if not req_data or 'suhu' not in req_data or 'kelembapan' not in req_data:
        return jsonify({'pesan': 'request gagal'}), 400

    new_id = len(data) + 1
    new_data = {
        'id': new_id,
        'suhu': req_data['suhu'],
        'kelembapan': req_data['kelembapan']
    }
    data.append(new_data)
    return jsonify({'pesan': 'Data sukses diterima', 'data': new_data}), 201

if __name__ == '__main__':
    app.run(debug=True, port=6969)

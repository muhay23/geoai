from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, firestore, messaging
import numpy as np
from sklearn.cluster import DBSCAN
import os
import json
from datetime import datetime

app = Flask(__name__)

# ── CONNECT TO FIREBASE ──────────────────────────────────────────────
service_account_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'))
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── PING ENDPOINT (for UptimeRobot keep-alive) ────────────────────────
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})


# ── HELPER: GET ALL USER FCM TOKENS ──────────────────────────────────
def get_all_fcm_tokens():
    tokens = []
    try:
        users = db.collection('users').stream()
        for user in users:
            data = user.to_dict()
            token = data.get('fcmToken')
            if token:
                tokens.append(token)
    except Exception as e:
        print(f'Failed to get FCM tokens: {e}')
    return tokens


# ── HELPER: SEND NOTIFICATION TO ALL USERS ───────────────────────────
def send_notification_to_all(title, body):
    try:
        tokens = get_all_fcm_tokens()
        if not tokens:
            print('No FCM tokens found')
            return

        batch_size = 500
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                android=messaging.AndroidConfig(
                    priority='high',
                    notification=messaging.AndroidNotification(
                        channel_id='conflict_tracker_channel',
                        priority='high',
                    ),
                ),
                tokens=batch,
            )
            response = messaging.send_each_for_multicast(message)
            print(f'Sent {response.success_count} notifications successfully')
            if response.failure_count > 0:
                print(f'Failed to send {response.failure_count} notifications')

    except Exception as e:
        print(f'Failed to send notifications: {e}')


# ── HELPER: SEND NOTIFICATION TO ONE USER ────────────────────────────
def send_notification_to_user(token, title, body):
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    channel_id='conflict_tracker_channel',
                    priority='high',
                ),
            ),
            token=token,
        )
        response = messaging.send(message)
        print(f'Sent notification: {response}')
    except Exception as e:
        print(f'Failed to send notification to user: {e}')


# ── ENDPOINT: NOTIFY REPORT STATUS CHANGE ────────────────────────────
@app.route('/notify_report', methods=['POST'])
def notify_report():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        status = data.get('status')
        category = data.get('category', 'incident')

        if not user_id or not status:
            return jsonify({'success': False, 'error': 'Missing userId or status'}), 400

        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        token = user_doc.to_dict().get('fcmToken')
        if not token:
            return jsonify({'success': False, 'error': 'No FCM token for user'}), 404

        if status == 'approved':
            title = '✅ Report Approved'
            body = f'Your {category} report has been reviewed and approved by our team.'
        else:
            title = '❌ Report Rejected'
            body = f'Your {category} report was reviewed but could not be verified at this time.'

        send_notification_to_user(token, title, body)

        return jsonify({'success': True, 'message': f'Notification sent to user {user_id}'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── MAIN ANALYZE ENDPOINT ─────────────────────────────────────────────
@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        reports_ref = db.collection('reports').where('status', '==', 'approved')
        docs = list(reports_ref.stream())

        incidents = []
        for doc in docs:
            data = doc.to_dict()
            lat = data.get('latitude') or data.get('latitude ')
            lng = data.get('longitude') or data.get('longitude ')
            if lat is None or lng is None:
                continue
            incidents.append({
                'lat': float(lat),
                'lng': float(lng),
                'severity': data.get('severity', 'low'),
                'category': data.get('category', 'unknown'),
                'description': data.get('description', ''),
            })

        if len(incidents) == 0:
            return jsonify({
                'success': True,
                'zones': [],
                'message': 'No approved incidents found'
            })

        zones = []
        now_iso = datetime.utcnow().isoformat()

        if len(incidents) >= 2:
            coords = np.array([[i['lat'], i['lng']] for i in incidents])
            coords_rad = np.radians(coords)

            kms_per_radian = 6371.0088
            epsilon = 1.0 / kms_per_radian

            labels = DBSCAN(
                eps=epsilon,
                min_samples=2,
                algorithm='ball_tree',
                metric='haversine'
            ).fit(coords_rad).labels_

            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue

                cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
                cluster_incidents = [incidents[i] for i in cluster_indices]

                center_lat = float(np.mean([i['lat'] for i in cluster_incidents]))
                center_lng = float(np.mean([i['lng'] for i in cluster_incidents]))

                max_dist = 0
                for i in cluster_incidents:
                    dist = np.sqrt(
                        (i['lat'] - center_lat) ** 2 +
                        (i['lng'] - center_lng) ** 2
                    ) * 111
                    if dist > max_dist:
                        max_dist = dist
                radius_km = max(max_dist * 1.3, 0.5)

                severities = [i['severity'] for i in cluster_incidents]
                if 'high' in severities:
                    severity = 'high'
                elif 'medium' in severities:
                    severity = 'medium'
                else:
                    severity = 'low'

                categories = list(set([i['category'] for i in cluster_incidents]))
                cat_str = ', '.join(categories)

                description = (
                    f"{len(cluster_incidents)} incident(s) detected in this area. "
                    f"Categories: {cat_str}. "
                    f"Severity: {severity.upper()}. "
                    f"Civilians advised to avoid this zone."
                )

                zones.append({
                    'centerLat': center_lat,
                    'centerLng': center_lng,
                    'radiusKm': radius_km,
                    'severity': severity,
                    'incidentCount': len(cluster_incidents),
                    'description': description,
                    'categories': categories,
                    'active': True,
                    'zoneType': 'cluster',
                    'createdAt': now_iso,
                })

            for i, label in enumerate(labels):
                if label == -1 and incidents[i]['severity'] == 'high':
                    inc = incidents[i]
                    zones.append({
                        'centerLat': inc['lat'],
                        'centerLng': inc['lng'],
                        'radiusKm': 0.5,
                        'severity': 'high',
                        'incidentCount': 1,
                        'description': (
                            f"High severity incident detected. "
                            f"Category: {inc['category']}. "
                            f"Severity: HIGH. "
                            f"Civilians advised to avoid this zone."
                        ),
                        'categories': [inc['category']],
                        'active': True,
                        'zoneType': 'single',
                        'createdAt': now_iso,
                    })

        else:
            inc = incidents[0]
            if inc['severity'] == 'high':
                zones.append({
                    'centerLat': inc['lat'],
                    'centerLng': inc['lng'],
                    'radiusKm': 0.5,
                    'severity': 'high',
                    'incidentCount': 1,
                    'description': (
                        f"High severity incident detected. "
                        f"Category: {inc['category']}. "
                        f"Severity: HIGH. "
                        f"Civilians advised to avoid this zone."
                    ),
                    'categories': [inc['category']],
                    'active': True,
                    'zoneType': 'single',
                    'createdAt': now_iso,
                })

        # ── CLEAR OLD ZONES AND SAVE NEW ONES ────────────────────────
        old_zones = db.collection('danger_zones').stream()
        for z in old_zones:
            z.reference.delete()

        for zone in zones:
            db.collection('danger_zones').add(zone)

    # ── SEND NOTIFICATION ONLY FOR HIGH AND MEDIUM SEVERITY ──────────
high_medium_zones = [z for z in zones if z['severity'] in ['high', 'medium']]

if len(high_medium_zones) > 0:
    high_count = len([z for z in high_medium_zones if z['severity'] == 'high'])
    medium_count = len([z for z in high_medium_zones if z['severity'] == 'medium'])

    if high_count > 0:
        title = '🚨 Danger Zone Alert'
        body = (f'{high_count} HIGH severity danger zone(s) detected. '
                f'Stay safe and avoid affected areas.')
    else:
        title = '⚠️ Danger Zone Warning'
        body = (f'{medium_count} MEDIUM severity danger zone(s) detected. '
                f'Exercise caution and stay informed.')

    send_notification_to_all(title, body)

        return jsonify({
            'success': True,
            'zones': zones,
            'message': f'{len(zones)} danger zone(s) identified'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
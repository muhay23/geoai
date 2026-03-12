from flask import Flask, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import os
import json

app = Flask(__name__)

# ── CONNECT TO FIREBASE ──────────────────────────────────────────────
service_account_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'))
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        # ── FETCH APPROVED INCIDENTS ─────────────────────────────────
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
            })

        if len(incidents) < 2:
            return jsonify({
                'success': True,
                'zones': [],
                'message': 'Not enough incidents to form zones'
            })

        # ── RUN DBSCAN CLUSTERING ─────────────────────────────────────
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

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # ── BUILD ZONES ───────────────────────────────────────────────
        zones = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_incidents = [incidents[i] for i, l in enumerate(labels) if l == cluster_id]

            center_lat = float(np.mean([i['lat'] for i in cluster_incidents]))
            center_lng = float(np.mean([i['lng'] for i in cluster_incidents]))

            max_dist = 0
            for i in cluster_incidents:
                dist = np.sqrt((i['lat'] - center_lat)**2 + (i['lng'] - center_lng)**2) * 111
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

            zone_data = {
                'centerLat': center_lat,
                'centerLng': center_lng,
                'radiusKm': radius_km,
                'severity': severity,
                'incidentCount': len(cluster_incidents),
                'description': description,
                'categories': categories,
                'active': True,
            }

            zones.append(zone_data)

        # ── CLEAR OLD ZONES AND SAVE NEW ONES ────────────────────────
        old_zones = db.collection('danger_zones').stream()
        for z in old_zones:
            z.reference.delete()

        for zone in zones:
            db.collection('danger_zones').add(zone)

        return jsonify({
            'success': True,
            'zones': zones,
            'message': f'{len(zones)} danger zones identified'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
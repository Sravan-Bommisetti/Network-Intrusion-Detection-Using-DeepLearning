import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploaded_files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ── Load models ───────────────────────────────────────────────────────────────
knn_bin   = pickle.load(open('knn_binary_class.sav', 'rb'))
knn_multi = pickle.load(open('knn_multi_class.sav', 'rb'))
cnn_bin   = tf.keras.models.load_model('latest_cnn_bin.h5')
cnn_multi = tf.keras.models.load_model('latest_cnn_multiclass.h5')

# ── Mappings ──────────────────────────────────────────────────────────────────
PROT_MAP = {'tcp': 1, 'udp': 2, 'icmp': 0}
SERVICE_MAP = {
    'IRC':0,'X11':1,'Z39_50':2,'http_8001':3,'auth':4,'bgp':5,'courier':6,
    'csnet_ns':7,'ctf':8,'daytime':9,'discard':10,'domain':11,'domain_u':12,
    'echo':13,'eco_i':14,'ecr_i':15,'efs':16,'exec':17,'finger':18,'ftp':19,
    'ftp_data':20,'gopher':21,'harvest':22,'hostnames':23,'http':24,
    'http_2784':25,'http_443':26,'aol':27,'imap4':28,'iso_tsap':29,
    'klogin':30,'kshell':31,'ldap':32,'link':33,'login':34,'mtp':35,
    'name':36,'netbios_dgm':37,'netbios_ns':38,'netbios_ssn':39,'netstat':40,
    'nnsp':41,'nntp':42,'ntp_u':43,'other':44,'pm_dump':45,'pop_2':46,
    'pop_3':47,'printer':48,'private':49,'red_i':50,'remote_job':51,
    'rje':52,'shell':53,'smtp':54,'sql_net':55,'ssh':56,'sunrpc':57,
    'supdup':58,'systat':59,'telnet':60,'tftp_u':61,'tim_i':62,'time':63,
    'urh_i':64,'urp_i':65,'uucp':66,'uucp_path':67,'vmnet':68,'whois':69
}
FLAG_MAP = {
    'OTH':0,'REJ':1,'RSTO':2,'RSTOS0':3,'RSTR':4,
    'S0':5,'S1':6,'S2':7,'S3':8,'SF':9,'SH':10
}
ATTACK_DESC = {
    'dos':   ('DoS',   'A Denial-of-Service (DoS) attack floods a target to make it unavailable to users.'),
    'probe': ('Probe', 'Probing scans network devices for weaknesses to exploit later.'),
    'r2l':   ('R2L',   'Remote to Local: an attacker sends packets to a machine where they have no local account.'),
    'u2r':   ('U2R',   'User to Root: attacker escalates from normal user to full root access.'),
    'normal':('Normal','Traffic is safe.'),
}
CNN_IDX = {0:'dos', 1:'normal', 2:'probe', 3:'r2l', 4:'u2r'}


def _describe(attack_key):
    ak = attack_key.lower() if attack_key else 'normal'
    return ATTACK_DESC.get(ak, ('Unknown', 'No description available.'))


def _predict_on_row(row_df):
    """Run KNN and CNN on a single-row DataFrame."""
    results = {}

    # KNN
    bin_knn = knn_bin.predict(row_df)[0]
    multi_knn = knn_multi.predict(row_df)[0]
    label, desc = _describe(multi_knn)
    results['knn'] = {
        'binary': 'Attack' if bin_knn == 1 else 'Normal',
        'multi': label,
        'description': desc,
    }

    # CNN
    tp1 = Normalizer().fit_transform(row_df.values)
    bin_cnn = cnn_bin.predict(np.reshape(tp1, (tp1.shape[0], 1, tp1.shape[1])), verbose=0)
    bin_cnn_val = round(float(bin_cnn[0][0]))
    multi_cnn = cnn_multi.predict(np.reshape(tp1, (tp1.shape[0], tp1.shape[1], 1)), verbose=0)
    cnn_idx = [round(float(v)) for v in multi_cnn[0]]
    cnn_attack = next((CNN_IDX[i] for i, v in enumerate(cnn_idx) if v == 1), 'normal')
    label, desc = _describe(cnn_attack)
    results['cnn'] = {
        'binary': 'Attack' if bin_cnn_val == 1 else 'Normal',
        'multi': label,
        'description': desc,
    }

    return results


def _load_and_scale_csv(path):
    cols = ['protocol_type','service','flag','logged_in','count',
            'srv_serror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
            'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_serror_rate','dst_host_rerror_rate']
    df = pd.read_csv(path)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    df['protocol_type'] = LabelEncoder().fit_transform(df['protocol_type'])
    df['service']       = LabelEncoder().fit_transform(df['service'])
    df['flag']          = LabelEncoder().fit_transform(df['flag'])
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    return scaled


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           services=sorted(SERVICE_MAP.keys()),
                           flags=sorted(FLAG_MAP.keys()))


@app.route('/predict/random', methods=['POST'])
def predict_random():
    data = pd.read_csv('fs_new validation project.csv')
    cols = ['protocol_type','service','flag','logged_in','count',
            'srv_serror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
            'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_serror_rate','dst_host_rerror_rate','attack']
    data.columns = cols
    data['protocol_type'] = LabelEncoder().fit_transform(data['protocol_type'])
    data['service']       = LabelEncoder().fit_transform(data['service'])
    data['flag']          = LabelEncoder().fit_transform(data['flag'])
    x = data.drop(['attack'], axis=1)
    x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)
    row = x_scaled.sample()
    return jsonify({'status': 'ok', 'results': _predict_on_row(row)})


@app.route('/predict/params', methods=['POST'])
def predict_params():
    d = request.get_json()
    row = [
        PROT_MAP.get(d['protocol_type'], 1),
        SERVICE_MAP.get(d['service'], 24),
        FLAG_MAP.get(d['flag'], 9),
        int(d['logged_in']),
        int(d['count']),
        float(d['srv_serror_rate']),
        float(d['srv_rerror_rate']),
        float(d['same_srv_rate']),
        float(d['diff_srv_rate']),
        int(d['dst_host_count']),
        int(d['dst_host_srv_count']),
        float(d['dst_host_same_srv_rate']),
        float(d['dst_host_diff_srv_rate']),
        float(d['dst_host_same_src_port_rate']),
        float(d['dst_host_serror_rate']),
        float(d['dst_host_rerror_rate']),
    ]
    cols = ['protocol_type','service','flag','logged_in','count',
            'srv_serror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
            'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_serror_rate','dst_host_rerror_rate']
    row_df = pd.DataFrame([row], columns=cols)
    return jsonify({'status': 'ok', 'results': _predict_on_row(row_df)})


@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'status': 'error', 'message': 'Empty filename'}), 400
    model_choice = request.form.get('model', 'knn')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    try:
        f.save(save_path)
        x = _load_and_scale_csv(save_path)
        rows = []

        if model_choice == 'knn':
            for b, m in zip(knn_bin.predict(x), knn_multi.predict(x)):
                label, _ = _describe(m)
                rows.append({'binary': 'Attack' if b == 1 else 'Normal', 'multi': label})

        elif model_choice == 'cnn':
            xv = Normalizer().fit_transform(x.values)
            bin_preds   = cnn_bin.predict(np.reshape(xv, (xv.shape[0], 1, xv.shape[1])), verbose=0)
            multi_preds = cnn_multi.predict(np.reshape(xv, (xv.shape[0], xv.shape[1], 1)), verbose=0)
            for b, m in zip(bin_preds, multi_preds):
                bv  = round(float(b[0]))
                idx = [round(float(v)) for v in m]
                ak  = next((CNN_IDX[i] for i, v in enumerate(idx) if v == 1), 'normal')
                label, _ = _describe(ak)
                rows.append({'binary': 'Attack' if bv == 1 else 'Normal', 'multi': label})

        summary = {
            'total':   len(rows),
            'attacks': sum(1 for r in rows if r['binary'] == 'Attack'),
            'normal':  sum(1 for r in rows if r['binary'] == 'Normal'),
        }
        return jsonify({'status': 'ok', 'summary': summary, 'rows': rows[:200]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)

from flask import Flask, render_template, jsonify, request
from agents.meta.self_awareness import SelfAwareness, InternalState
from knowledge.knowledge_base import KnowledgeBase, Knowledge
from learning.continuous_learning import ContinuousLearner, LearningTask
import json
import time
import uuid

app = Flask(__name__)

# 初始化系统组件
self_awareness = SelfAwareness()
knowledge_base = KnowledgeBase()
continuous_learner = ContinuousLearner()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    state = self_awareness.evaluate_self()
    return jsonify(state)

@app.route('/api/state', methods=['POST'])
def update_state():
    data = request.json
    state = InternalState(
        id=str(uuid.uuid4()),
        type=data['type'],
        value=float(data['value']),
        confidence=float(data['confidence']),
        source=data['source'],
        timestamp=time.time(),
        metadata=data.get('metadata', {})
    )
    success = self_awareness.update_state(state)
    return jsonify({'success': success})

@app.route('/api/knowledge', methods=['GET'])
def get_knowledge():
    query = request.args.get('query', '')
    domain = request.args.get('domain')
    knowledge = knowledge_base.search_knowledge(query, domain)
    return jsonify([{
        'id': k.id,
        'content': k.content,
        'type': k.type,
        'domain': k.domain,
        'confidence': k.confidence
    } for k in knowledge])

@app.route('/api/knowledge', methods=['POST'])
def add_knowledge():
    data = request.json
    knowledge = Knowledge(
        id=str(uuid.uuid4()),
        content=data['content'],
        type=data['type'],
        domain=data['domain'],
        confidence=float(data['confidence']),
        source=data['source'],
        timestamp=time.time(),
        last_updated=time.time(),
        metadata=data.get('metadata', {})
    )
    success = knowledge_base.add_knowledge(knowledge)
    return jsonify({'success': success})

@app.route('/api/learning', methods=['GET'])
def get_learning_status():
    tasks = continuous_learner.tasks
    return jsonify([{
        'id': task.id,
        'name': task.name,
        'type': task.type,
        'status': task.status,
        'metrics': task.metrics
    } for task in tasks.values()])

@app.route('/api/learning', methods=['POST'])
def create_learning_task():
    data = request.json
    task = LearningTask(
        id=str(uuid.uuid4()),
        name=data['name'],
        type=data['type'],
        data=data['data']
    )
    success = continuous_learner.create_task(task)
    return jsonify({'success': success, 'task_id': task.id})

if __name__ == '__main__':
    app.run(debug=True) 
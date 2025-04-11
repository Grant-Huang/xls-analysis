from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import os
from werkzeug.utils import secure_filename
import numpy as np
import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

def clean_data(df):
    """清理数据，处理NaN和空值"""
    # 将NaN替换为None，这样JSON序列化时会被转换为null
    df = df.replace({np.nan: None})
    # 将空字符串替换为None
    df = df.replace({'': None})
    return df

def safe_json_serialize(obj):
    """安全地序列化对象，处理特殊值"""
    if pd.isna(obj) or obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(obj, np.ndarray):
        return [safe_json_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(x) for x in obj]
    return str(obj)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_columns', methods=['POST'])
def get_columns():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_excel(filepath)
            print(f"成功读取文件，列名: {df.columns.tolist()}")
            return jsonify({'columns': df.columns.tolist()})
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/get_unique_values', methods=['POST'])
def get_unique_values():
    data = request.get_json()
    column = data.get('column')
    
    if not column:
        return jsonify({'error': '未指定列名'}), 400
    
    # 获取最新上传的文件
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not files:
        return jsonify({'error': '没有可用的数据文件'}), 400
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], latest_file)
    
    try:
        df = pd.read_excel(filepath)
        # 清理数据
        df = clean_data(df)
        # 获取唯一值并安全序列化
        values = [safe_json_serialize(v) for v in df[column].unique()]
        # 过滤掉None值
        values = [v for v in values if v is not None]
        print(f"获取列 {column} 的唯一值: {values[:5]}...")
        return jsonify({'values': values})
    except Exception as e:
        print(f"获取唯一值时出错: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    # 获取最新上传的文件
    upload_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    print(f"上传目录: {upload_dir}")
    
    files = os.listdir(upload_dir)
    if not files:
        print("上传目录为空")
        return jsonify({'error': '没有可用的数据文件'}), 400
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(upload_dir, x)))
    filepath = os.path.join(upload_dir, latest_file)
    print(f"使用文件: {filepath}")
    
    try:
        df = pd.read_excel(filepath)
        print(f"成功读取文件，数据形状: {df.shape}")
        
        # 清理数据
        df = clean_data(df)
        
        # 获取图表配置
        data = request.get_json()
        if not data:
            print("未收到图表配置数据")
            return jsonify({'error': '未收到图表配置数据'}), 400
            
        chart_type = data.get('chartType', 'bar')
        x_axis = data.get('xAxis')
        y_axis = data.get('yAxis')
        group_by = data.get('groupBy')
        filter_column = data.get('filterColumn')
        filter_value = data.get('filterValue')
        
        print(f"图表配置: 类型={chart_type}, X轴={x_axis}, Y轴={y_axis}, 分组={group_by}, 筛选={filter_column}={filter_value}")
        
        # 验证必填字段
        if not x_axis or not y_axis:
            return jsonify({'error': '请选择X轴和Y轴维度'}), 400
        
        # 应用筛选
        if filter_column and filter_value:
            df = df[df[filter_column] == filter_value]
            print(f"筛选后数据形状: {df.shape}")
        
        # 根据图表类型进行排序
        if chart_type in ['bar', 'line']:
            # 对于柱状图和折线图，按X轴排序
            df = df.sort_values(by=[x_axis])
        elif chart_type == 'pie':
            # 对于饼图，按Y轴值降序排序
            df = df.sort_values(by=[y_axis], ascending=False)
        elif chart_type == 'scatter':
            # 对于散点图，按X轴排序
            df = df.sort_values(by=[x_axis])
        
        # 根据图表类型创建图表
        if chart_type == 'bar':
            if group_by:
                fig = px.bar(df, x=x_axis, y=y_axis, color=group_by,
                           title=f'{y_axis}按{x_axis}分组统计',
                           labels={x_axis: x_axis, y_axis: y_axis, group_by: group_by})
            else:
                fig = px.bar(df, x=x_axis, y=y_axis,
                           title=f'{y_axis}按{x_axis}统计',
                           labels={x_axis: x_axis, y_axis: y_axis})
        elif chart_type == 'line':
            if group_by:
                fig = px.line(df, x=x_axis, y=y_axis, color=group_by,
                            title=f'{y_axis}按{x_axis}趋势',
                            labels={x_axis: x_axis, y_axis: y_axis, group_by: group_by})
            else:
                fig = px.line(df, x=x_axis, y=y_axis,
                            title=f'{y_axis}按{x_axis}趋势',
                            labels={x_axis: x_axis, y_axis: y_axis})
        elif chart_type == 'pie':
            fig = px.pie(df, values=y_axis, names=x_axis,
                       title=f'{y_axis}按{x_axis}分布',
                       labels={x_axis: x_axis, y_axis: y_axis})
        elif chart_type == 'scatter':
            if group_by:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=group_by,
                               title=f'{y_axis}与{x_axis}相关性',
                               labels={x_axis: x_axis, y_axis: y_axis, group_by: group_by})
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis,
                               title=f'{y_axis}与{x_axis}相关性',
                               labels={x_axis: x_axis, y_axis: y_axis})
        
        # 更新图表布局
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            legend_title=group_by if group_by else None,
            showlegend=True if group_by else False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # 打印图表数据
        print(f"图表数据: {fig.data}")
        print(f"图表布局: {fig.layout}")
        
        # 使用自定义序列化器处理图表数据
        chart_data = []
        for trace in fig.data:
            # 根据图表类型处理数据
            if chart_type == 'pie':
                trace_data = {
                    'type': trace.type,
                    'values': trace.values.tolist() if hasattr(trace.values, 'tolist') else trace.values,
                    'labels': trace.labels.tolist() if hasattr(trace.labels, 'tolist') else trace.labels,
                    'name': trace.name,
                    'textinfo': 'label+percent',
                    'hoverinfo': 'label+percent+value'
                }
            else:
                trace_data = {
                    'type': trace.type,
                    'x': trace.x.tolist() if hasattr(trace.x, 'tolist') else trace.x,
                    'y': trace.y.tolist() if hasattr(trace.y, 'tolist') else trace.y,
                    'name': trace.name,
                    'mode': trace.mode if hasattr(trace, 'mode') else None,
                    'marker': {
                        'color': trace.marker.color if hasattr(trace, 'marker') else None
                    }
                }
                if group_by:
                    trace_data['legendgroup'] = trace.legendgroup
            chart_data.append(trace_data)
        
        # 序列化布局
        chart_layout = {
            'title': fig.layout.title.text,
            'xaxis': {
                'title': fig.layout.xaxis.title.text if hasattr(fig.layout.xaxis, 'title') else None,
                'type': 'category' if chart_type in ['bar', 'pie'] else None
            },
            'yaxis': {
                'title': fig.layout.yaxis.title.text if hasattr(fig.layout.yaxis, 'title') else None
            },
            'showlegend': fig.layout.showlegend,
            'legend': {
                'title': fig.layout.legend.title.text if hasattr(fig.layout.legend, 'title') else None
            }
        }
        
        return jsonify({
            'data': chart_data,
            'layout': chart_layout
        })
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 
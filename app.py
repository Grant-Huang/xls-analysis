from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import os
from werkzeug.utils import secure_filename
import numpy as np
import datetime
import plotly.graph_objects as go

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 版本号
VERSION = '1.17'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

def clean_data(df):
    """清理数据，处理空值和非法数据"""
    # 将空字符串和None替换为NaN
    df = df.replace(['', 'None', 'null', 'NULL', 'Null'], np.nan)
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

def apply_filter_condition(df, filter_condition):
    column = filter_condition.get('column')
    operator = filter_condition.get('operator')
    value = filter_condition.get('value')
    
    if not all([column, operator, value]):
        return df
    
    try:
        if operator == '=':
            return df[df[column] == value]
        elif operator == '!=':
            return df[df[column] != value]
        elif operator == '>':
            return df[df[column] > float(value)]
        elif operator == '<':
            return df[df[column] < float(value)]
        elif operator == '>=':
            return df[df[column] >= float(value)]
        elif operator == '<=':
            return df[df[column] <= float(value)]
        elif operator == 'contains':
            return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == 'not_contains':
            return df[~df[column].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            print(f"不支持的运算符: {operator}")
            return df
    except Exception as e:
        print(f"应用筛选条件时出错: {str(e)}")
        return df

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        chart_type = data.get('chartType')
        x_axis = data.get('xAxis')
        y_axis = data.get('yAxis')
        group_by = data.get('groupBy')
        filter_conditions = data.get('filterConditions', [])
        z_axis = data.get('zAxis')
        flow_column = data.get('flowColumn')

        if not chart_type or not x_axis or not y_axis:
            return jsonify({'error': '缺少必要的参数'}), 400

        # 获取最新上传的文件
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not files:
            return jsonify({'error': '没有可用的数据文件'}), 400
            
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], latest_file)
        
        # 读取数据
        df = pd.read_excel(filepath)
        print(f"成功读取文件: {filepath}, 数据形状: {df.shape}")

        # 清理数据
        df = clean_data(df)

        # 应用筛选条件
        for condition in filter_conditions:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if not all([column, operator, value]):
                continue
                
            # 处理多个条件值
            if isinstance(value, str):
                values = [v.strip() for v in value.split(',')]
            elif isinstance(value, list):
                values = [str(v).strip() for v in value]
            else:
                values = [str(value).strip()]
                
            if not values:
                continue
                
            if operator == 'equals':
                df = df[df[column].isin(values)]
            elif operator == 'not_equals':
                df = df[~df[column].isin(values)]
            elif operator == 'contains':
                mask = pd.Series(False, index=df.index)
                for v in values:
                    mask |= df[column].astype(str).str.contains(v, case=False, na=False)
                df = df[mask]
            elif operator == 'not_contains':
                mask = pd.Series(True, index=df.index)
                for v in values:
                    mask &= ~df[column].astype(str).str.contains(v, case=False, na=False)
                df = df[mask]
            elif operator == 'greater_than':
                try:
                    values = [float(v) for v in values]
                    df = df[df[column] > min(values)]
                except (ValueError, TypeError):
                    continue
            elif operator == 'less_than':
                try:
                    values = [float(v) for v in values]
                    df = df[df[column] < max(values)]
                except (ValueError, TypeError):
                    continue
            elif operator == 'greater_than_or_equal':
                try:
                    values = [float(v) for v in values]
                    df = df[df[column] >= min(values)]
                except (ValueError, TypeError):
                    continue
            elif operator == 'less_than_or_equal':
                try:
                    values = [float(v) for v in values]
                    df = df[df[column] <= max(values)]
                except (ValueError, TypeError):
                    continue

        # 根据图表类型处理数据
        if chart_type in ['bar', 'line', 'area', 'scatter', 'box', 'radar', 'polar', 'funnel']:
            # 这些图表类型支持分组
            if group_by:
                df = df.groupby([x_axis, group_by])[y_axis].sum().reset_index()
            else:
                df = df.groupby(x_axis)[y_axis].sum().reset_index()
            df = df.sort_values(by=y_axis, ascending=False)
            
        elif chart_type == 'pie':
            # 饼图不支持分组
            df = df.groupby(x_axis)[y_axis].sum().reset_index()
            df = df.sort_values(by=y_axis, ascending=False)
            
        elif chart_type == 'heatmap':
            if not z_axis:
                return jsonify({'error': '请选择数值列'}), 400
            # 热力图不支持分组
            pivot_df = df.pivot_table(
                index=x_axis,
                columns=y_axis,
                values=z_axis,
                aggfunc='sum'
            )
            # 创建热力图
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values.tolist(),
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale='Viridis',
                text=pivot_df.values.tolist(),
                hoverongaps=False
            ))
            # 更新布局
            fig.update_layout(
                xaxis_title=y_axis,
                yaxis_title=x_axis,
                height=600
            )
            
        elif chart_type == 'sankey':
            if not flow_column:
                return jsonify({'error': '请选择源节点列、目标节点列和流量列'}), 400
            # 桑基图不支持分组
            df = df.groupby([x_axis, y_axis])[flow_column].sum().reset_index()
            # 创建桑基图数据
            nodes = list(set(df[x_axis].tolist() + df[y_axis].tolist()))
            node_dict = {node: i for i, node in enumerate(nodes)}
            source = df[x_axis].map(node_dict).tolist()
            target = df[y_axis].map(node_dict).tolist()
            value = df[flow_column].tolist()
            
            trace = {
                'type': 'sankey',
                'node': {
                    'label': nodes,
                    'pad': 15,
                    'thickness': 20,
                    'line': {
                        'color': 'black',
                        'width': 0.5
                    }
                },
                'link': {
                    'source': source,
                    'target': target,
                    'value': value
                }
            }
            
            layout = {
                'title': {
                    'text': f'从{x_axis}到{y_axis}的流向分析'
                },
                'height': 600,
                'font': {
                    'size': 10
                }
            }
            
            return jsonify({
                'data': [trace],
                'layout': layout
            })
            
        elif chart_type == 'parallel':
            parallel_column = data.get('parallelColumn')
            if not parallel_column:
                return jsonify({'error': '请选择第三个维度'}), 400
            # 平行坐标图使用专门的第三个维度列
            df = df[[x_axis, y_axis, parallel_column]]
            
        elif chart_type == 'bubble':
            size_column = data.get('sizeColumn')
            if not size_column:
                return jsonify({'error': '请选择气泡大小列'}), 400
            # 气泡图不支持分组，使用专门的大小列
            df = df[[x_axis, y_axis, size_column]]
            
            # 确保数值列是数值类型
            try:
                df[size_column] = pd.to_numeric(df[size_column], errors='coerce')
                # 移除无效值
                df = df.dropna(subset=[size_column])
                if df.empty:
                    return jsonify({'error': '气泡大小列包含无效的数值'}), 400
            except Exception as e:
                return jsonify({'error': f'处理气泡大小列时出错: {str(e)}'}), 400

            # 创建气泡图
            fig = go.Figure()
            size_values = df[size_column].tolist()
            size_values = [float(x) for x in size_values if pd.notna(x)]
            
            if not size_values:
                return jsonify({'error': '气泡大小列没有有效的数值'}), 400
                
            fig.add_trace(go.Scatter(
                x=df[x_axis].tolist(),
                y=df[y_axis].tolist(),
                mode='markers',
                marker=dict(
                    size=size_values,
                    sizemode='area',
                    sizeref=2.*max(size_values)/(40.**2),
                    sizemin=4
                )
            ))
                
        elif chart_type == 'treemap':
            parent_column = data.get('parentColumn')
            # 树图使用专门的父级列
            if parent_column:
                df = df[[x_axis, y_axis, parent_column]]
            else:
                df = df[[x_axis, y_axis]]

        # 创建图表
        fig = go.Figure()
        
        if chart_type == 'bar':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Bar(
                        x=group_df[x_axis].tolist(),
                        y=group_df[y_axis].tolist(),
                        name=str(group)
                    ))
            else:
                fig.add_trace(go.Bar(
                    x=df[x_axis].tolist(),
                    y=df[y_axis].tolist()
                ))
                
        elif chart_type == 'line':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Scatter(
                        x=group_df[x_axis].tolist(),
                        y=group_df[y_axis].tolist(),
                        name=str(group),
                        mode='lines+markers'
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[x_axis].tolist(),
                    y=df[y_axis].tolist(),
                    mode='lines+markers'
                ))
                
        elif chart_type == 'area':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Scatter(
                        x=group_df[x_axis].tolist(),
                        y=group_df[y_axis].tolist(),
                        name=str(group),
                        mode='lines',
                        fill='tonexty'
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[x_axis].tolist(),
                    y=df[y_axis].tolist(),
                    mode='lines',
                    fill='tozeroy'
                ))
                
        elif chart_type == 'pie':
            fig.add_trace(go.Pie(
                labels=df[x_axis].tolist(),
                values=df[y_axis].tolist(),
                textinfo='label+percent',
                hoverinfo='label+value'
            ))
            
        elif chart_type == 'scatter':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Scatter(
                        x=group_df[x_axis].tolist(),
                        y=group_df[y_axis].tolist(),
                        name=str(group),
                        mode='markers'
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[x_axis].tolist(),
                    y=df[y_axis].tolist(),
                    mode='markers'
                ))
                
        elif chart_type == 'bubble':
            size_column = data.get('sizeColumn')
            # 确保所有值都是有效的数值
            size_values = df[size_column].tolist()
            size_values = [float(x) if pd.notna(x) else 0 for x in size_values]
            
            fig.add_trace(go.Scatter(
                x=df[x_axis].tolist(),
                y=df[y_axis].tolist(),
                mode='markers',
                marker=dict(
                    size=size_values,
                    sizemode='area',
                    sizeref=2.*max(size_values)/(40.**2),
                    sizemin=4
                )
            ))
                
        elif chart_type == 'box':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Box(
                        y=group_df[y_axis].tolist(),
                        name=str(group)
                    ))
            else:
                fig.add_trace(go.Box(
                    y=df[y_axis].tolist()
                ))
                
        elif chart_type == 'radar':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Scatterpolar(
                        r=group_df[y_axis].tolist(),
                        theta=group_df[x_axis].tolist(),
                        name=str(group),
                        fill='toself'
                    ))
            else:
                fig.add_trace(go.Scatterpolar(
                    r=df[y_axis].tolist(),
                    theta=df[x_axis].tolist(),
                    fill='toself'
                ))
                
        elif chart_type == 'polar':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Barpolar(
                        r=group_df[y_axis].tolist(),
                        theta=group_df[x_axis].tolist(),
                        name=str(group)
                    ))
            else:
                fig.add_trace(go.Barpolar(
                    r=df[y_axis].tolist(),
                    theta=df[x_axis].tolist()
                ))
                
        elif chart_type == 'funnel':
            if group_by:
                for group in df[group_by].unique():
                    group_df = df[df[group_by] == group]
                    fig.add_trace(go.Funnel(
                        y=group_df[x_axis].tolist(),
                        x=group_df[y_axis].tolist(),
                        name=str(group)
                    ))
            else:
                fig.add_trace(go.Funnel(
                    y=df[x_axis].tolist(),
                    x=df[y_axis].tolist()
                ))
                
        elif chart_type == 'treemap':
            parent_column = data.get('parentColumn')
            if parent_column:
                fig.add_trace(go.Treemap(
                    labels=df[x_axis].tolist(),
                    parents=df[parent_column].tolist(),
                    values=df[y_axis].tolist()
                ))
            else:
                fig.add_trace(go.Treemap(
                    labels=df[x_axis].tolist(),
                    values=df[y_axis].tolist()
                ))
                
        elif chart_type == 'parallel':
            parallel_column = data.get('parallelColumn')
            fig.add_trace(go.Parcoords(
                line=dict(
                    color=df[parallel_column].tolist(),
                    colorscale='Viridis'
                ),
                dimensions=[
                    dict(range=[df[x_axis].min(), df[x_axis].max()],
                         label=x_axis, values=df[x_axis].tolist()),
                    dict(range=[df[y_axis].min(), df[y_axis].max()],
                         label=y_axis, values=df[y_axis].tolist()),
                    dict(range=[df[parallel_column].min(), df[parallel_column].max()],
                         label=parallel_column, values=df[parallel_column].tolist())
                ]
            ))

        # 更新布局
        fig.update_layout(
            title=f'{chart_type.capitalize()} Chart',
            showlegend=True,
            height=600
        )

        # 序列化图表数据
        chart_data = {
            'data': [{
                'type': trace.type,
                'x': [safe_json_serialize(x) for x in (trace.x if hasattr(trace, 'x') else [])],
                'y': [safe_json_serialize(x) for x in (trace.y if hasattr(trace, 'y') else [])],
                'labels': [safe_json_serialize(x) for x in (trace.labels if hasattr(trace, 'labels') else [])],
                'values': [safe_json_serialize(x) for x in (trace.values if hasattr(trace, 'values') else [])],
                'name': safe_json_serialize(trace.name) if hasattr(trace, 'name') else None,
                'mode': trace.mode if hasattr(trace, 'mode') else None,
                'marker': safe_json_serialize(trace.marker.to_plotly_json() if hasattr(trace, 'marker') else {}),
                'textinfo': trace.textinfo if hasattr(trace, 'textinfo') else None,
                'hoverinfo': trace.hoverinfo if hasattr(trace, 'hoverinfo') else None,
                'fill': trace.fill if hasattr(trace, 'fill') else None,
                'node': safe_json_serialize(trace.node.to_plotly_json() if hasattr(trace, 'node') else {}),
                'link': safe_json_serialize(trace.link.to_plotly_json() if hasattr(trace, 'link') else {}),
                'dimensions': [safe_json_serialize(x) for x in (trace.dimensions if hasattr(trace, 'dimensions') else [])],
                'line': safe_json_serialize(trace.line.to_plotly_json() if hasattr(trace, 'line') else {})
            } for trace in fig.data],
            'layout': {
                'title': {'text': safe_json_serialize(fig.layout.title.text if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text') else f'{chart_type.capitalize()} Chart')},
                'showlegend': fig.layout.showlegend if hasattr(fig.layout, 'showlegend') else True,
                'height': fig.layout.height if hasattr(fig.layout, 'height') else 600,
                'xaxis': {
                    'title': {'text': safe_json_serialize(fig.layout.xaxis.title.text if hasattr(fig.layout, 'xaxis') and hasattr(fig.layout.xaxis, 'title') and hasattr(fig.layout.xaxis.title, 'text') else x_axis)}
                },
                'yaxis': {
                    'title': {'text': safe_json_serialize(fig.layout.yaxis.title.text if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title') and hasattr(fig.layout.yaxis.title, 'text') else y_axis)}
                }
            }
        }

        return jsonify(chart_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
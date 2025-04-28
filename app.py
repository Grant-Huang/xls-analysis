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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

def clean_data(df):
    """清理数据，处理空值和非法数据"""
    # 将空字符串和None替换为NaN
    df = df.replace(['', 'None', 'null', 'NULL', 'Null', '-'], np.nan)
    
    # 定义数值列和分类列
    numeric_columns = ['免税总额', '已交付数量', '已开票数量', '总计', '订单年份', '订单月份', '订购数量']
    categorical_columns = ['产品变体', '产品变体.1', '类别', '尺寸', '材料与花色', '发票状态', '客户', '币种', '相关订单', '销售人员']
    
    # 处理数值列
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                continue
    
    # 处理分类列
    for col in categorical_columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('nan', np.nan)
            except Exception as e:
                continue
    
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
        # 生成唯一的文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"文件已保存: {filepath}")
            
            df = pd.read_excel(filepath)
            print(f"成功读取文件，列名: {df.columns.tolist()}")
            return jsonify({
                'columns': df.columns.tolist(),
                'filename': filename  # 返回文件名供后续使用
            })
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/get_unique_values', methods=['POST'])
def get_unique_values():
    data = request.get_json()
    column = data.get('column')
    filename = data.get('filename')  # 获取文件名
    
    print(f"接收到的请求数据: {data}")
    print(f"当前文件名: {filename}")
    
    if not column:
        return jsonify({'error': '未指定列名'}), 400
    
    if not filename:
        return jsonify({'error': '未找到数据文件'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"文件路径: {filepath}")
    print(f"文件是否存在: {os.path.exists(filepath)}")
    
    if not os.path.exists(filepath):
        return jsonify({'error': '数据文件不存在'}), 400
    
    try:
        df = pd.read_excel(filepath)
        print(f"成功读取文件，数据形状: {df.shape}")
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
    """应用筛选条件到数据框"""
    try:
        column = filter_condition.get('column')
        operator = filter_condition.get('operator')
        value = filter_condition.get('value')
        
        if not all([column, operator, value]):
            return df
            
        # 确保列存在
        if column not in df.columns:
            return df
            
        # 处理数值比较
        if operator in ['>', '<', '>=', '<=', '=']:
            try:
                # 尝试将值转换为数值类型
                value = float(value)
                # 确保列是数值类型
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except (ValueError, TypeError):
                return df
                
            if operator == '>':
                df = df[df[column] > value]
            elif operator == '<':
                df = df[df[column] < value]
            elif operator == '>=':
                df = df[df[column] >= value]
            elif operator == '<=':
                df = df[df[column] <= value]
            elif operator == '=':
                df = df[df[column] == value]
                
        # 处理字符串比较
        elif operator in ['contains', 'not_contains']:
            if isinstance(value, list):
                if operator == 'contains':
                    mask = df[column].astype(str).str.contains('|'.join(value), case=False, na=False)
                else:
                    mask = ~df[column].astype(str).str.contains('|'.join(value), case=False, na=False)
                df = df[mask]
            else:
                if operator == 'contains':
                    df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
                else:
                    df = df[~df[column].astype(str).str.contains(str(value), case=False, na=False)]
                    
        return df
        
    except Exception as e:
        return df

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        
        # 获取基本参数
        chart_type = data.get('chartType')
        x_axis = data.get('xAxis')
        y_axis = data.get('yAxis')
        group_by = data.get('groupBy')
        filter_conditions = data.get('filterConditions', [])
        filename = data.get('filename')
        size_column = data.get('sizeColumn')

        # 验证基本参数
        if not all([chart_type, x_axis, y_axis, filename]):
            return jsonify({'error': '缺少必要的参数'}), 400

        # 获取文件路径
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '数据文件不存在'}), 400

        # 读取数据
        try:
            df = pd.read_excel(filepath)
            
            # 清理数据
            df = clean_data(df)
            
            # 应用筛选条件
            for condition in filter_conditions:
                df = apply_filter_condition(df, condition)
            
            if df.empty:
                return jsonify({'error': '筛选后的数据为空，请检查筛选条件'}), 400
            
            # 根据图表类型处理数据
            if chart_type == 'pie':
                # 确保数值列是数值类型
                try:
                    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
                    # 删除包含NaN的行
                    df = df.dropna(subset=[x_axis, y_axis])
                    
                    if df.empty:
                        return jsonify({'error': '数据中包含无效的数值，请检查选择的列'}), 400
                except Exception as e:
                    print(f"转换数据时出错: {str(e)}")
                    return jsonify({'error': f'处理数据时出错: {str(e)}'}), 400
                
                # 根据分组处理数据
                if group_by:
                    df = df.groupby([x_axis, group_by])[y_axis].sum().reset_index()
                    # 创建饼图
                    fig = go.Figure()
                    for group in df[group_by].unique():
                        group_df = df[df[group_by] == group]
                        fig.add_trace(go.Pie(
                            labels=group_df[x_axis],
                            values=group_df[y_axis],
                            name=str(group),
                            hole=0.3
                        ))
                else:
                    df = df.groupby(x_axis)[y_axis].sum().reset_index()
                    # 创建饼图
                    fig = go.Figure(data=[go.Pie(
                        labels=df[x_axis],
                        values=df[y_axis],
                        hole=0.3
                    )])
                
                # 更新布局
                fig.update_layout(
                    title='饼图',
                    showlegend=True,
                    height=600
                )
                
                # 返回图表数据
                return jsonify({
                    'data': [{
                        'type': 'pie',
                        'labels': df[x_axis].tolist(),
                        'values': df[y_axis].tolist(),
                        'name': group_by if group_by else None,
                        'hole': 0.3
                    }],
                    'layout': {
                        'title': {'text': '饼图'},
                        'showlegend': True,
                        'height': 600
                    }
                })
            elif chart_type == 'bubble':
                # 确保X轴、Y轴和大小列都是数值类型
                try:
                    df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
                    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
                    df[size_column] = pd.to_numeric(df[size_column], errors='coerce')
                    
                    # 删除包含NaN的行
                    df = df.dropna(subset=[x_axis, y_axis, size_column])
                    
                    if df.empty:
                        return jsonify({'error': '数据中包含无效的数值，请检查选择的列'}), 400
                except Exception as e:
                    print(f"转换数据时出错: {str(e)}")
                    return jsonify({'error': f'处理数据时出错: {str(e)}'}), 400
                
                # 创建气泡图
                fig = go.Figure()
                
                if group_by:
                    for group in df[group_by].unique():
                        group_df = df[df[group_by] == group]
                        fig.add_trace(go.Scatter(
                            x=group_df[x_axis],
                            y=group_df[y_axis],
                            mode='markers',
                            marker=dict(
                                size=group_df[size_column],
                                sizemode='area',
                                sizeref=2.*max(group_df[size_column])/(40.**2),
                                sizemin=4
                            ),
                            name=str(group)
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df[x_axis],
                        y=df[y_axis],
                        mode='markers',
                        marker=dict(
                            size=df[size_column],
                            sizemode='area',
                            sizeref=2.*max(df[size_column])/(40.**2),
                            sizemin=4
                        )
                    ))
                
                # 更新布局
                fig.update_layout(
                    title='气泡图',
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    showlegend=True if group_by else False,
                    height=600
                )
                
                # 返回图表数据
                return jsonify({
                    'data': [{
                        'type': 'scatter',
                        'x': df[x_axis].tolist(),
                        'y': df[y_axis].tolist(),
                        'mode': 'markers',
                        'marker': {
                            'size': df[size_column].tolist(),
                            'sizemode': 'area',
                            'sizeref': 2.*max(df[size_column])/(40.**2),
                            'sizemin': 4
                        },
                        'name': group_by if group_by else None
                    }],
                    'layout': {
                        'title': {'text': '气泡图'},
                        'xaxis': {'title': {'text': x_axis}},
                        'yaxis': {'title': {'text': y_axis}},
                        'showlegend': True if group_by else False,
                        'height': 600
                    }
                })
            elif chart_type == 'sankey':
                # 确保源节点和目标节点列是有效的分类数据
                source_col = data.get('xAxis')
                target_col = data.get('yAxis')
                flow_col = data.get('flowColumn')
                
                if not all([source_col, target_col, flow_col]):
                    return jsonify({'error': '桑基图需要源节点列、目标节点列和流量列'})
                
                # 将源节点和目标节点列转换为字符串类型
                df[source_col] = df[source_col].astype(str)
                df[target_col] = df[target_col].astype(str)
                
                # 移除空值
                df = df.dropna(subset=[source_col, target_col, flow_col])
                
                # 确保流量列是数值类型
                df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
                df = df.dropna(subset=[flow_col])
                
                if df.empty:
                    return jsonify({'error': '没有有效的数据可以生成桑基图'})
                
                # 获取所有唯一的节点
                all_nodes = list(set(df[source_col].unique()) | set(df[target_col].unique()))
                
                # 为每个节点分配唯一的颜色
                node_colors = {}
                for i, node in enumerate(all_nodes):
                    # 使用HSL颜色空间，确保颜色分布均匀
                    hue = (i * 360) / len(all_nodes)
                    node_colors[node] = f'hsl({hue}, 70%, 50%)'
                
                # 创建桑基图数据
                sankey_data = {
                    'type': 'sankey',
                    'orientation': 'h',
                    'node': {
                        'pad': 15,
                        'thickness': 30,
                        'line': {
                            'color': "black",
                            'width': 0.5
                        },
                        'label': all_nodes,
                        'color': [node_colors[node] for node in all_nodes]
                    },
                    'link': {
                        'source': [all_nodes.index(source) for source in df[source_col]],
                        'target': [all_nodes.index(target) for target in df[target_col]],
                        'value': df[flow_col].tolist(),
                        'color': [node_colors[source] for source in df[source_col]]
                    }
                }
                
                return jsonify({
                    'data': [sankey_data],
                    'layout': {
                        'title': {
                            'text': f'从 {source_col} 到 {target_col} 的流向分析'
                        },
                        'height': 600
                    }
                })
            elif chart_type == 'bar':
                # 确保X轴和Y轴列存在
                x_col = data.get('xAxis')
                y_col = data.get('yAxis')
                
                if not all([x_col, y_col]):
                    return jsonify({'error': '柱状图需要X轴和Y轴列'})
                
                # 确保Y轴是数值类型
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                
                # 处理分组
                group_col = data.get('groupBy')
                if group_col:
                    # 将分组列转换为字符串类型，并将 NaN 替换为 "未知"
                    df[group_col] = df[group_col].astype(str)
                    df[group_col] = df[group_col].replace('nan', '未知')
                    
                    # 按分组和X轴列进行分组，计算Y轴值的总和
                    grouped_df = df.groupby([x_col, group_col])[y_col].sum().reset_index()
                    
                    # 获取所有唯一的分组值
                    groups = sorted(grouped_df[group_col].unique())
                    
                    # 为每个分组分配唯一的颜色
                    group_colors = {}
                    for i, group in enumerate(groups):
                        # 使用HSL颜色空间，确保颜色分布均匀
                        hue = (i * 360) / len(groups)
                        group_colors[group] = f'hsl({hue}, 70%, 50%)'
                    
                    # 按分组创建数据
                    traces = []
                    for group in groups:
                        group_df = grouped_df[grouped_df[group_col] == group]
                        # 确保数据中没有 NaN 值
                        group_df = group_df.dropna()
                        if not group_df.empty:
                            traces.append({
                                'type': 'bar',
                                'x': group_df[x_col].tolist(),
                                'y': group_df[y_col].tolist(),
                                'name': str(group),
                                'marker': {
                                    'color': group_colors[group]
                                }
                            })
                else:
                    # 没有分组时，按X轴列进行分组
                    grouped_df = df.groupby(x_col)[y_col].sum().reset_index()
                    grouped_df = grouped_df.dropna()
                    
                    # 为每个X轴值分配不同的颜色
                    x_values = grouped_df[x_col].unique()
                    x_colors = {}
                    for i, x in enumerate(x_values):
                        hue = (i * 360) / len(x_values)
                        x_colors[x] = f'hsl({hue}, 70%, 50%)'
                    
                    traces = [{
                        'type': 'bar',
                        'x': grouped_df[x_col].tolist(),
                        'y': grouped_df[y_col].tolist(),
                        'marker': {
                            'color': [x_colors[x] for x in grouped_df[x_col]]
                        }
                    }]
                
                if not traces:
                    return jsonify({'error': '没有有效的数据可以生成柱状图'})
                
                return jsonify({
                    'data': traces,
                    'layout': {
                        'title': {
                            'text': f'{y_col} 在 {x_col} 维度上的分布'
                        },
                        'showlegend': bool(group_col),
                        'height': 600,
                        'xaxis': {
                            'title': {
                                'text': x_col
                            }
                        },
                        'yaxis': {
                            'title': {
                                'text': y_col
                            }
                        }
                    }
                })
            elif chart_type == 'line':
                # 确保X轴和Y轴列存在
                x_col = data.get('xAxis')
                y_col = data.get('yAxis')
                
                if not all([x_col, y_col]):
                    return jsonify({'error': '折线图需要X轴和Y轴列'})
                
                # 确保Y轴是数值类型
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                
                # 处理分组
                group_col = data.get('groupBy')
                if group_col:
                    # 将分组列转换为字符串类型，并将 NaN 替换为 "未知"
                    df[group_col] = df[group_col].astype(str)
                    df[group_col] = df[group_col].replace('nan', '未知')
                    
                    # 按分组和X轴列进行分组，计算Y轴值的总和
                    grouped_df = df.groupby([x_col, group_col])[y_col].sum().reset_index()
                    
                    # 获取所有唯一的分组值
                    groups = sorted(grouped_df[group_col].unique())
                    
                    # 为每个分组分配唯一的颜色
                    group_colors = {}
                    for i, group in enumerate(groups):
                        # 使用HSL颜色空间，确保颜色分布均匀
                        hue = (i * 360) / len(groups)
                        group_colors[group] = f'hsl({hue}, 70%, 50%)'
                    
                    # 按分组创建数据
                    traces = []
                    for group in groups:
                        group_df = grouped_df[grouped_df[group_col] == group]
                        # 确保数据中没有 NaN 值
                        group_df = group_df.dropna()
                        if not group_df.empty:
                            traces.append({
                                'type': 'scatter',
                                'mode': 'lines+markers',
                                'x': group_df[x_col].tolist(),
                                'y': group_df[y_col].tolist(),
                                'name': str(group),
                                'line': {
                                    'color': group_colors[group],
                                    'width': 2
                                },
                                'marker': {
                                    'color': group_colors[group],
                                    'size': 8
                                }
                            })
                else:
                    # 没有分组时，按X轴列进行分组
                    grouped_df = df.groupby(x_col)[y_col].sum().reset_index()
                    grouped_df = grouped_df.dropna()
                    
                    # 为折线图使用单一颜色
                    traces = [{
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'x': grouped_df[x_col].tolist(),
                        'y': grouped_df[y_col].tolist(),
                        'line': {
                            'color': 'rgb(31, 119, 180)',
                            'width': 2
                        },
                        'marker': {
                            'color': 'rgb(31, 119, 180)',
                            'size': 8
                        }
                    }]
                
                if not traces:
                    return jsonify({'error': '没有有效的数据可以生成折线图'})
                
                return jsonify({
                    'data': traces,
                    'layout': {
                        'title': {
                            'text': f'{y_col} 随 {x_col} 的变化趋势'
                        },
                        'showlegend': bool(group_col),
                        'height': 600,
                        'xaxis': {
                            'title': {
                                'text': x_col
                            }
                        },
                        'yaxis': {
                            'title': {
                                'text': y_col
                            }
                        }
                    }
                })
            else:
                # 根据图表类型处理数据
                if group_by:
                    df = df.groupby([x_axis, group_by])[y_axis].sum().reset_index()
                else:
                    df = df.groupby(x_axis)[y_axis].sum().reset_index()
                
                print(f"分组后的数据形状: {df.shape}")
                print(f"最终数据示例:\n{df.head()}")
                
                # 创建图表
                fig = go.Figure()
                
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

                # 更新布局
                fig.update_layout(
                    title=f'{chart_type.capitalize()} Chart',
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    showlegend=True if group_by else False,
                    height=600
                )

                # 返回图表数据
                return jsonify({
                    'data': [{
                        'type': 'bar',
                        'x': df[x_axis].tolist(),
                        'y': df[y_axis].tolist(),
                        'name': group_by if group_by else None
                    }],
                    'layout': {
                        'title': {'text': f'{chart_type.capitalize()} Chart'},
                        'xaxis': {'title': {'text': x_axis}},
                        'yaxis': {'title': {'text': y_axis}},
                        'showlegend': True if group_by else False,
                        'height': 600
                    }
                })

        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            return jsonify({'error': f'处理数据时出错: {str(e)}'}), 400

    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 
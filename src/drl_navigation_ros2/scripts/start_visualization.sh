#!/bin/bash

# 多机器人训练可视化启动脚本
# 此脚本会启动 RViz2 和 TensorBoard

echo "🚀 启动多机器人训练可视化工具"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# RViz2 配置文件路径
RVIZ_CONFIG="$SCRIPT_DIR/../config/multi_robot_visualization_3r.rviz"

# TensorBoard 日志目录
TENSORBOARD_LOGDIR="$PROJECT_ROOT/runs"

echo "📊 项目根目录: $PROJECT_ROOT"
echo "📊 RViz2 配置: $RVIZ_CONFIG"
echo "📊 TensorBoard 日志: $TENSORBOARD_LOGDIR"
echo ""

# 检查 RViz2 配置文件是否存在
if [ ! -f "$RVIZ_CONFIG" ]; then
    echo "❌ 错误: RViz2 配置文件不存在: $RVIZ_CONFIG"
    exit 1
fi

# 询问用户要启动什么
echo "请选择要启动的可视化工具:"
echo "  1) RViz2 (机器人模型、激光雷达、目标点)"
echo "  2) TensorBoard (训练曲线)"
echo "  3) 同时启动 RViz2 和 TensorBoard"
echo ""
read -p "请输入选项 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🎨 启动 RViz2..."
        rviz2 -d "$RVIZ_CONFIG"
        ;;
    2)
        echo ""
        echo "📈 启动 TensorBoard..."
        echo "   访问地址: http://localhost:6006"
        tensorboard --logdir="$TENSORBOARD_LOGDIR" --port=6006
        ;;
    3)
        echo "🎨 启动 RViz2 (后台)..."
        rviz2 -d "$RVIZ_CONFIG" &
        RVIZ_PID=$!
        
        echo "📈 启动 TensorBoard..."
        echo "   访问地址: http://localhost:6006"
        echo ""
        echo "💡 提示: 按 Ctrl+C 停止 TensorBoard 和 RViz2"
        
        # 设置清理函数
        cleanup() {
            echo ""
            echo "🛑 正在停止可视化工具..."
            kill $RVIZ_PID 2>/dev/null
            echo "✅ 已停止"
            exit 0
        }
        
        trap cleanup INT TERM
        
        tensorboard --logdir="$TENSORBOARD_LOGDIR" --port=6006
        ;;
    *)
        echo "❌ 无效的选项"
        exit 1
        ;;
esac


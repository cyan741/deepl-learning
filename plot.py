import matplotlib.pyplot as plt
def loss_curve_visulization(train_loss, train_acc, val_loss, val_acc):
    # 画出损失函数图像
    fig, ax1 = plt.subplots()

    # 绘制训练损失曲线
    ax1.plot(train_loss, color='tab:blue', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建第二个y轴对象
    ax2 = ax1.twinx()

    # 绘制训练准确率曲线
    ax2.plot(train_acc, color='tab:orange', label='Train Accuracy')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 添加验证损失和准确率曲线
    ax1.plot(val_loss, linestyle='--', color='tab:blue', label='Validation Loss')
    ax2.plot(val_acc, linestyle='--', color='tab:orange', label='Validation Accuracy')

    # 添加图例
    fig.legend(loc='upper right')

    # 显示图形
    plt.title('Training and Validation Metrics')
    plt.savefig("./figs/image.png")
    plt.show()
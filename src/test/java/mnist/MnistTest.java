package mnist;

import mnist.utils.MnistUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * @Author huangweiye
 * @Date 2024/12/26
 * @Description
 */
public class MnistTest {

    private static Logger logger = LoggerFactory.getLogger(MnistTest.class);
    public static void main(String[] args) throws Exception {
        // 定义输入图片的行和列（每张图片的大小为 28x28）
        final int numRows = 28;
        final int numColumns = 28;

        // 输出类别数量（MNIST 数据集包含数字 0 到 9，共 10 类）
        int outputNum = 10;

        // 每次训练时处理的批量样本数
        int batchSize = 128;

        // 随机数种子，用于确保结果的可重复性
        int rngSeed = 123;

        // 训练的总周期数
        int numEpochs = 15;

        logger.info("构建模型....");

        // 配置神经网络的架构
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)  // 设置随机种子，确保训练结果可重复
                .updater(new Nesterovs(0.006, 0.9))  // 使用 Nesterov 动量优化器，学习率为 0.006，动量为 0.9
                .l2(1e-4)  // 使用 L2 正则化，防止过拟合
                .list()  // 开始定义网络的层次
                .layer(new DenseLayer.Builder()  // 第一层：全连接层（输入层）
                        .nIn(numRows * numColumns)  // 输入层节点数：28x28 的图片展平为 784 个输入节点
                        .nOut(1000)  // 隐藏层节点数：设定为 1000
                        .activation(Activation.RELU)  // 激活函数：ReLU
                        .weightInit(WeightInit.XAVIER)  // 权重初始化方法：Xavier 初始化
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)  // 第二层：输出层
                        .nIn(1000)  // 输入节点数：1000 个隐藏层节点
                        .nOut(outputNum)  // 输出节点数：10 个类别（对应数字 0-9）
                        .activation(Activation.SOFTMAX)  // 激活函数：Softmax，用于多类分类
                        .weightInit(WeightInit.XAVIER)  // 权重初始化方法：Xavier 初始化
                        .build())
                .build();  // 构建模型配置

        // 创建神经网络模型
        MultiLayerNetwork trainModel = new MultiLayerNetwork(conf);
        trainModel.init();  // 初始化模型

        // 设置训练过程中每一轮的得分输出，1 表示每完成 1 次迭代输出一次得分
        trainModel.setListeners(new ScoreIterationListener(1));

        logger.info("开始训练模型....");

        // 使用训练数据集训练模型，训练 15 个 epoch（训练周期）
        // 获取训练集和测试集的数据迭代器
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        trainModel.fit(mnistTrain, numEpochs);

        logger.info("评估模型....");

        // 使用测试数据集评估训练好的模型
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        Evaluation eval = trainModel.evaluate(mnistTest);
        logger.info(eval.stats());  // 输出评估结果，包括准确率、精确率等指标

        // 保存模型到文件
        String model = "D:\\idea-work\\dl4j-example\\src\\test\\resources\\model\\mnist\\mnist_model.zip";
        File modelFile = new File(model);
        ModelSerializer.writeModel(trainModel, modelFile, true);
        logger.info("****************训练示例完成********************");

        // 加载已训练的模型
        MultiLayerNetwork testModel = MultiLayerNetwork.load(new File(model), true);
        // 测试图像路径
        String testImagePath = "D:\\idea-work\\dl4j-example\\src\\test\\resources\\model-data\\mnist\\";
        // 假设你有10个测试图像，命名为 0.png 到 9.png，当我从MNIST数据集网站下载9张图片后，这个大模型确实可以给我识别出来
        for (int i = 0; i < 3; i++) {
            String fileName = testImagePath + i + ".jpg";
            System.out.println("fileName=" + fileName);
            INDArray testImage = MnistUtils.loadGrayImg(fileName);
            INDArray output = testModel.output(testImage); // 进行预测

            // 获取预测结果
            int predictedClass = Nd4j.argMax(output, 1).getInt(0);
            System.out.println("测试图像 " + i + " 的预测结果: " + predictedClass);
        }
    }

}

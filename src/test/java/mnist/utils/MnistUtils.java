package mnist.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @Author huangweiye
 * @Date 2025/2/8
 * @Description
 */
public class MnistUtils {
    /**
     * 将图像转换为灰度图像
     *
     * @param original
     * @return
     */
    private static BufferedImage convertToGrayscale(BufferedImage original) {
        BufferedImage grayImage = new BufferedImage(original.getWidth(), original.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = grayImage.getGraphics();
        g.drawImage(original, 0, 0, null);
        g.dispose();
        return grayImage;
    }

    /**
     * 调整图像大小
     *
     * @param original
     * @param width
     * @param height
     * @return
     */
    private static BufferedImage resizeImage(BufferedImage original, int width, int height) {
        Image scaledImage = original.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(scaledImage, 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }

    /**
     * 加载图像
     *
     * @param fileName
     * @return
     */
    public static INDArray loadGrayImg(String fileName) {
        try {
            // 1. 加载图片
            BufferedImage originalImage = ImageIO.read(new File(fileName));
            // 2. 转换为灰度图像
            BufferedImage grayImage = convertToGrayscale(originalImage);
            // 3. 调整大小为 28x28 像素
            BufferedImage resizedImage = resizeImage(grayImage, 28, 28);
            // 4. 进行归一化处理
            return normalizeImage(resizedImage);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * 对图像进行归一化处理并生成 INDArray
     *
     * @param image
     * @return
     */
    private static INDArray normalizeImage(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[] normalizedData = new double[width * height]; // 创建一维数组

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 获取灰度值（0-255）
                int grayValue = image.getRGB(x, y) & 0xFF; // 只取灰度部分
                // 归一化到 [0, 1] 范围
                normalizedData[y * width + x] = grayValue / 255.0; // 填充一维数组
            }
        }

        // 将一维数组转换为 INDArray，并添加批次维度
        INDArray indArray = Nd4j.create(normalizedData).reshape(1, 28*28); // reshape to [1, 784] 28*28 是 特征数量（图片宽高）
        return indArray;
    }

}

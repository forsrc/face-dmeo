package com.forsrc.facedemo;

import java.io.IOException;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.core.io.ClassPathResource;

public class Demo {

	private static CascadeClassifier cascadeClassifier;
	static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.loadLibrary("opencv_java451");
        try {
			cascadeClassifier = new CascadeClassifier(new ClassPathResource("haarcascade_frontalface_alt.xml").getFile().getAbsolutePath());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	

	public static void main(String[] args) throws IOException {
		String image1 = new ClassPathResource("/2.png").getFile().getAbsolutePath();
		String image2 = new ClassPathResource("/1.png").getFile().getAbsolutePath();
		//System.out.println(image1);
		Mat mat_1 = toMat(image1);
		Mat mat_2 = toMat(image2);
		Mat hist_1 = new Mat();
		Mat hist_2 = new Mat();

		MatOfFloat ranges = new MatOfFloat(0f, 256f);

		MatOfInt histSize = new MatOfInt(1000);

		Imgproc.calcHist(Arrays.asList(mat_1), new MatOfInt(0), new Mat(), hist_1, histSize, ranges);
		Imgproc.calcHist(Arrays.asList(mat_2), new MatOfInt(0), new Mat(), hist_2, histSize, ranges);

		double res = Imgproc.compareHist(hist_1, hist_2, Imgproc.CV_COMP_CORREL);

		System.out.println(res);

	}

	public static Mat toMat(String img) {
		Mat image0 = Imgcodecs.imread(img);

		Mat image1 = new Mat();

		Imgproc.cvtColor(image0, image1, Imgproc.COLOR_BGR2GRAY);

		MatOfRect faceDetections = new MatOfRect();
		cascadeClassifier.detectMultiScale(image1, faceDetections);

		for (Rect rect : faceDetections.toArray()) {
			Mat face = new Mat(image1, rect);
			return face;
		}
		return null;
	}

}

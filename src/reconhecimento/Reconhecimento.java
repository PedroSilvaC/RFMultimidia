/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reconhecimento;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

/**
 *
 * @author POSITIVO
 */
public class Reconhecimento {
    public static void main(String[] args) throws FrameGrabber.Exception {
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        String[] pessoas = {"", "wanderson", "pedro", "railson", "lucas"};
        camera.start();
        
        CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\08-haarcascade-frontalface-alt.xml");
        System.out.println("testando");
        
        //FaceRecognizer reconhecedor = createEigenFaceRecognizer();
        //reconhecedor.load("src\\recursos\\classificadorEigenFaces.yml");
        FaceRecognizer reconhecedor = createFisherFaceRecognizer();
        reconhecedor.load("src\\recursos\\classificadorFisherFaces.yml");
        
        System.out.println("test");
        CanvasFrame cFrame = new CanvasFrame("Reconhecimento", CanvasFrame.getDefaultGamma() / camera.getGamma());//faz automaticamente o desenho de uma janela
        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();
        
        while((frameCapturado = camera.grab()) != null){//enquanto tiver capturando da webcam, faça
            imagemColorida = converteMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGR2GRAY);
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 2, 0, new Size(150, 150), new Size(500, 500));
            System.out.println("testinggg");
            for (int i=0; i < facesDetectadas.size(); i++){
                System.out.println("teste");
                Rect dadosFace = facesDetectadas.get(i);//vai pegar primeira face que conseguir detectar e jogar na variavel 
                rectangle(imagemColorida, dadosFace, new Scalar(255, 255, 255, 0));
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);// vai jogar pra dentro faceCapturada somente dadosFace(aquela parte que tah dentro do quandrado
                resize(faceCapturada, faceCapturada, new Size(160, 160));//metodo pra imagens ter o msm tamanho pra não ter erros com os algortimos
                System.out.println("teste 2");
                IntPointer rotulo = new IntPointer(1);// classe p/a indicar se o rotulo 1 ou rotulo 2
                DoublePointer confianca = new DoublePointer();
                reconhecedor.predict(faceCapturada, rotulo, confianca);
                int predicao = rotulo.get(0);//resposta final
                System.out.println("predição:" + predicao);
                String nome;
                if(predicao == -1){
                    nome = "desconhecido";
                }else{
                    nome = pessoas[predicao] + " - " + confianca.get(0);
                }
                
                int x = Math.max(dadosFace.tl().x() - 10, 0);
                int y = Math.max(dadosFace.tl().y() - 10, 0);
                putText(imagemColorida, nome, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new Scalar(180, 180, 180, 0)); 
                
                
            }
            if(cFrame.isVisible()){
                cFrame.showImage(frameCapturado);
            }
        }
        cFrame.dispose();//liberar memoria da janela
        camera.stop();//parar a captura     
    }
}

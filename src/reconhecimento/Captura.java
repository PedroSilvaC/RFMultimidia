
package reconhecimento;


import java.awt.event.KeyEvent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
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
public class Captura {
    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException, FileNotFoundException, IOException {
       KeyEvent tecla = null;
       //convertMat será usado para converter frames em matrizes
       OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
       OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
       camera.start();
       
       //definimos nosso algoritmo detector de faces(haar cascade frontal face).
       CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\08-haarcascade-frontalface-alt.xml"); 
       //cFrame é a nossa janela de execulção da web cam, a nossa biblioteca já disponibiliza uma interface para isso, apenas passamos os parametros.
       CanvasFrame cFrame = new CanvasFrame("Previsualização", CanvasFrame.getDefaultGamma() / camera.getGamma());
       Frame frameCapturado = null; //Formato Frame = Pixels.
       Mat imagemColorida = new Mat();//Formato Mat = Matriz.
       
       int numeroDeAmostras = 25;//Quantidade de fotos de cada pessoa.
       int amostra = 1;//Amostra atual, contador auxiliar.
       int idPessoa; //Numero de identificação individual
       Scanner cadastro = new Scanner(System.in);
        
        System.out.println("informe seu nome");
        String nome = cadastro.nextLine();//Recebendo nome do usuário.
        
        File dirPastas = new File("src\\fotos");
        File dirFotos = new File("src\\fotos\\" + nome);
        //Analizando se já existe o diretorio
        if(!(dirFotos.isDirectory() && dirFotos.exists())){
            System.out.println("criando novo diretorio");
            idPessoa = (dirPastas.list().length) + 1;
            dirFotos.mkdir();
        }else{
            System.out.println("diretorio existente");
            //Configurando valores de amostra atual e id do usuário.
            String nomeFoto = dirFotos.list()[1];
            idPessoa = Integer.parseInt(nomeFoto.split(";")[1]);
            int amostraAtual = (dirFotos.list().length);
            amostra = amostraAtual + 1;
        }
            //Executando captura dos frames.
       while((frameCapturado = camera.grab()) != null){
           //Convertendo frameCapturado para matriz, e atribuindo à imagemColorida.
           imagemColorida = convertMat.convert(frameCapturado);
           Mat imagemCinza = new Mat();
           //Convertendo imagemColorida para ton de cinza.
           cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
           RectVector facesDetectadas = new RectVector();
           //Detectando face.
           detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150,150), new Size(500,500));
           if(tecla == null){
               tecla = cFrame.waitKey(5);
           }
           for (int i=0; i<facesDetectadas.size(); i++){
               //Rect classe modelo para retangulos 2d.
               Rect dadosFace = facesDetectadas.get(0);
               //Configurando nosso retangulo, principalmente cor.
               rectangle(imagemColorida, dadosFace, new Scalar(255, 0, 0, 0));
               Mat faceCapturada = new Mat(imagemCinza, dadosFace);
               //resize redimenciona a face capturada para um outro tamanho passado por parametro.
               resize(faceCapturada, faceCapturada, new Size(160,160));
               //Iniciando captura das fotos.
               if(tecla == null){
               tecla = cFrame.waitKey(5);
           }
               if (tecla != null){
                   if (tecla.getKeyChar() == 'q'){
                       if (amostra <= numeroDeAmostras){
                           //imwrite salva a foto capturada, no diretorio passado por parametro
                           imwrite("src\\fotos\\" + nome + "\\fotos." + nome + ";" + idPessoa + ";" + amostra + ".jpg", faceCapturada);
                           System.out.println("Foto " + amostra + " capturadao\n");
                           amostra++;
                       }   
                   }
                   tecla = null;
               }
                   
           }
           if(tecla == null){
               tecla = cFrame.waitKey(20);
           }
                
           if (cFrame.isVisible()){
               cFrame.showImage(frameCapturado);
           }
           if(amostra > numeroDeAmostras){
               break;
               
           
           }
       
       }
       //Desativando webCam.
       cFrame.dispose();
       camera.stop();
    }
}

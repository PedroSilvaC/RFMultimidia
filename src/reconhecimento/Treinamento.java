/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reconhecimento;

import java.io.File;
import java.nio.IntBuffer;
import java.util.ArrayList;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 *
 * @author POSITIVO
 */
public class Treinamento {
    public static void main(String[] args) {
        File diretorio = new File("src\\fotos");
        File[] vetorPastas = diretorio.listFiles();//Contém todas as pastas, ou seja, todos os usuários cadastrados.
        ArrayList<File> todasFotos = new ArrayList<>();// Array irá armazenar todas as fotos para o treinamento.
        //Deleta diretórios com número insulficiente de fotos.
        
        for (File file : vetorPastas){
            System.out.println(file);
            
            if(file.listFiles().length < 25){          //
                                                       //
                File[] arqApagar = file.listFiles();   // Aqui deletamos diretorios
                for(File file1 : arqApagar){           // com quantidade de fotos
                    file1.delete();                    // inferiores à 25
                }                                      //
                file.delete();                         //
            }
            
        }
        //Abaixo unimos todas as fotos em um unico ArrayList (todasFotos).
        for (File file: vetorPastas){
            for (File foto : file.listFiles()) {
                todasFotos.add(foto);
            }
        }
        
        MatVector fotos = new MatVector(todasFotos.size());//Formato MatVector = vetor de matrizes.
        //Rotulos são informações exenciais para o reconhecimento, essencial passar em formato CV_32SCI.
        Mat rotulos = new Mat(todasFotos.size(), 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        
        for (File imagem:todasFotos){
            //imread lê a foto no endereço da imagem atual. CV_LOAD_IMAGE_GRAYSCALE trabalha a imagem com escala de cinza.
            Mat foto = imread(imagem.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split(";")[1]);//Extraindo do nome da imagem o id do usuário.
            resize(foto, foto, new Size(160,160));//Redimencionando a imagem.
            
                                                 // ** Aprendizado Supervisionado **
            fotos.put(contador, foto);           //       -passando uma foto
            rotulosBuffer.put(contador, classe); //       relacionada à um id
            contador++;

        }
        

        
        
    FaceRecognizer eigenfaces = createEigenFaceRecognizer();
    FaceRecognizer fisherfaces = createFisherFaceRecognizer();
    FaceRecognizer lbph = createLBPHFaceRecognizer();
    
    eigenfaces.train(fotos, rotulos);
    eigenfaces.save("src\\recursos\\classificadorEigenFaces.yml");
   
    fisherfaces.train(fotos, rotulos);
    fisherfaces.save("src\\recursos\\classificadorFisherFaces.yml");
    
    lbph.train(fotos, rotulos);
    lbph.save("src\\recursos\\classificadorLBHP.yml");
            }
        }
    


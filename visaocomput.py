import cv2
import numpy as np

def detectar_objetos():

    
    url = "http://192.168.0.179:8080/video"
    cap = cv2.VideoCapture(url)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Erro ao capturar imagem da câmera")
            break
        
        height, width = frame.shape[:2]
        part_width = width // 3
        
        # Desenha as linhas divisórias 
        cv2.line(frame, (part_width, 0), (part_width, height), (0, 0, 255), 2)  # Linha vermelha
        cv2.line(frame, (part_width*2, 0), (part_width*2, height), (0, 255, 0), 2)  # Linha verde
        
        # Converte para HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara pro azul
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Máscara pro preto
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 0])  # Valor (V) baixo para preto
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Função para verificar posição e imprimir
        def verificar_posicao(contours, color_name):
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    
                    if cx < part_width:
                        print(f"{color_name} esquerda")
                    elif cx < part_width * 2:
                        print(f"{color_name} centro")
                    else:
                        print(f"{color_name} direita")
        
        # Verifica azul
        verificar_posicao(contours_blue, "azul")
        
        # Verifica preto
        verificar_posicao(contours_black, "preto")
        
        # Mostra a imagem e as máscaras 
        cv2.imshow('Camera', frame)
        cv2.imshow('Mascara Azul', mask_blue)
        cv2.imshow('Mascara Preta', mask_black)
        
        # Pressione 'q' para sair ou control C no terminal
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_objetos()
function [ porc ] = reconocimiento( X,U_K,k,K )
%%% Autor: Omar Pardo 130013 %%%
%%% Materia: Modelos Matemáticos, ITAM 2015 %%%
%%% Fecha: 18/08/2015 %%%

% Descripción.-
% Esta función toma los datos y la matriz de cada dígito, hace el pronóstico para
% cada vector, y calcula el porcentaje de acierto por dígito

% Input.-
% X: matriz de vectores con toda la muestra a pronosticar
% U_K: matriz que contiene las respectivas matrices U_K para cada dígito
% K: el número de valores singulares que se desean usar para el pronóstico

% Output.-
% porc: vector con el porcentaje de acierto para cada dígito, usando K valores singulares

% Iniciamos el vector porc
porc = zeros(10,1);

for l=0:9
    % Para cada dígito, iniciamos el contador de pronósticos correctos
    cont = 0;
    % Recorremos los 500 vectores de cada dígito
    for j=1:500
        r = zeros(10,1);
        % Definimos el vector a pronosticar
        z = X(:,l*500+j);
        for d=0:9
            % Comparamos con la matriz U_K de cada dígito
            UKd = U_K(:, d*K+1:d*K+k);
            % Calculamos el error, de pronosticar ese dígito
            r(d+1) = norm(z-UKd*UKd'*z);
        end
        % Obtenemos el dígito que dio el menor error
        [M,I] = min(r);
        % Si el pronóstico fue correcto, sumamos uno al contador
        if I-1 == l
        cont = cont+1;
        end
        j
    end
    % Dividimos el número de aciertos entre el número de pronósticos
    % y obtenemos la tasa de correctos
    porc(l+1)=cont/500;
end



end


function [ Udig ] = MUK( X,K )
%%% Autor: Omar Pardo 130013 %%%
%%% Materia: Modelos Matem�ticos, ITAM 2015 %%%
%%% Fecha: 18/08/2015 %%%

% Descripci�n.-
% Esta funci�n toma la matriz de datos y devuelve una matriz que contiene
% las respectivas matrices U_K para cada d�gito, tomando K vectores

% Input.-
% X: matriz de vectores con los d�gitos
% K: valores singulares a tomar

% Output.-
% Udig: matriz U de la DVS, tomando los primeros K vectores

% Iniciamos la matriz Udig
Udig = [];


for i = 0:9
    % Para cada d�gito tomamos su matriz de datos
    A = X(:, i*500+1:i*500+500);
    % Obtenemos su factorizaci�n DVS
    [U,S,V] = svd(A);
    % De la matriz U, tomamos s�lo los primeros K vectores
    U_K = U(:,1:K);
    % Los asignamos a la matriz de salida
    Udig = [Udig, U_K];
end

end


%%% Autor: Omar Pardo 130013 %%%
%%% Materia: Modelos Matemáticos, ITAM 2015 %%%
%%% Fecha: 18/08/2015 %%%

% Descripción.-
% Este script toma la muestra de los datos MNIST, usa la DVS para pronósticar dígitos
% y calcula el porcentaje de pronósticos correctos para cada dígito usando distintos valores
% de k para el número de valores singulares a usar.


% Cargamos la base de datos
load('data_numbers.mat');
X = X';

% Definimos el máximo número de valores singulares a usar
K = 1;

% Obtenemos la matriz U_K para cada dígito, con la K definida anteriormente
U_K = MUK(X,K);

% Iniciamos la matriz P, que contendrá el porcentaje de acierto por dígito y por k de 1 a K
P = zeros(10,K);

% Obtenemos el vector de aciertos para todos los dígitos, variando la k
for k = 1:K
    P(:,k) = reconocimiento(X,U_K,k,K);
end

% Regresamos la P
P









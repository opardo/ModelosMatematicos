function  ipm_mehrotra();
%--------------------------------------------------------------------------
%... Una prueba del código de puntos interiores extendido a la resolución
% de problemas de programación cuadrática 
%
%                                       JL Morales 2013
%                                       ITAM
%--------------------------------------------------------------------------

mu = 1.0e3;     % ... penalización para el termino lineal
TOL = 1.0e-8;   % ... tolerancia para la brecha de dualidad promedio

% Cargamos los datos
[T,test,ntrain,ntest] = wdbcData('wdbc.data',30,0.0,1);

% Obtenemos las dimensiones y quitamos el identificador
[n_row, n_col] = size(T);
n_atr = 30;
X = T(1:n_row,2:n_atr+1);

% Escalamos la matriz y la transponemos para dejarla en términos del modelo
[ X ] = scale(X);
X = X';

m = 0; k = 0;

% Separamos los tumores que resultaron benignos de los malignos, y
% obtenemos la información de las variables predictoras
for i=1:n_row
   if T(i,1) == 1
       m = m + 1;
       M(m,:) = T(i,2:31);
   else
       k = k + 1;
       B(k,:) = T(i,2:31);
   end
end

% Obtenemos el número de tumores total y el número de variables del módelo
mm = m + k;
nn = 2*n_atr + 2 + m + k + m + k;

fprintf(' Number of samples     ..........  %3i  \n', n_row);
fprintf(' Number of atributes   ..........  %3i  \n', n_atr);
fprintf(' Number of malignant samples ....  %3i  \n', m);
fprintf(' Number of benign samples .......  %3i  \n', k);
fprintf(' Number of variables      .......  %3i  \n', nn);
fprintf(' Number of constraints    .......  %3i  \n', mm);
fprintf(' Penalty parameter  .............  %8.2e  \n', mu);

e_m = ones(m,1); 
e_k = ones(k,1);

% Construimos la formulación primal del problema

%        w     \gamma
A = [  M -M  -e_m  e_m    eye(m)    zeros(m,k)   -eye(m)   zeros(m,k)  ;
      -B  B   e_k -e_k  zeros(k,m)    eye(k)    zeros(k,m)   -eye(k)   ];
  
b = [ e_m ; e_k ];
c = [ zeros(n_atr,1) ; zeros(n_atr,1) ; 0 ; 0 ; e_m  ; e_k ; zeros(m,1) ; zeros(k,1) ];
c = mu*c;

G = [ eye(n_atr) -eye(n_atr)  ;
      -eye(n_atr)  eye(n_atr) ];
  
Q = [            G                     zeros(2*n_atr, nn - 2*n_atr)      ;
      zeros(nn - 2*n_atr, 2*n_atr)   zeros( nn - 2*n_atr, nn - 2*n_atr ) ];

% Aprovechamos la estructura rala de las matrices 
A = sparse(A);
Q = sparse(Q);

tic

% Le aplicamos puntos interiores con sigma dinámico
[ x, lambda, s ] = ipm_second_order ( A, b, c, Q, TOL, 2 );

% Le aplicamos puntos interiores con sigma estático
[ x, lambda, s ] = ipm_second_order ( A, b, c, Q, TOL, 1 );

toc
%
%--------------------------------------------------------------------------
%
function [train,test,ntrain,ntest] = wdbcData(datafile,dataDim,fracTest,reord)
% syntax: [train,test,ntrain,ntest] = wdbcData(datafile,dataDim,fracTest,reord)
% extract data from the database
% here, "datafile" should be a string, eg 'wdbc.data'
%       "dataDim" is a scalar, eg 30
%       "fracTest" is a scalar strictly between 0 and 1 indicating
%       what fraction of data should go in the test set (the
%       remaining data goes in the training set)
%       "reord" indicates whether the data should be reordered
%       before selecting the test/training sets. Value of "0"
%       indicates no reordering, "1" indicates random reordering
%
% on return, ntrain and ntest indicate the number of rows in the
% training and testing sets, respectively. train and test contain the
% data arrays. Elements in first column of these matrices are either 0
% (indicating a benign sample) or 1 (indicating a malignant
% sample). Elements in the remaining columns, starting with column
% 2, contain the features.


% first check input data
if (nargin < 3)
    error('three or four input arguments are required for wdbcData');
end
if (~ischar(datafile))
    error('first argument must be a string');
end
if (isnumeric(dataDim)&max(size(dataDim))==1)
    if (isempty(dataDim))
        error('second argument must be a scalar');
    elseif (dataDim <= 0)
        error('second argument must be a positive integer');
    end
else
    error('second argument must be a positive number');
end
if(isnumeric(fracTest))
%     if(fracTest<=0 | fracTest>=1)
%         error('third argument must be a number strictly between 0 and 1');
%     end
else
    error('third argument must be a number');
end

if nargin==4
    if(~isnumeric(reord))
        error('fourth argument should be numeric, either 0 or 1');
    end
else
    % default is no reordering
    reord = 0;
end
reord = 0;



fp = fopen(datafile,'r');
samples = 0;
train = zeros(0,dataDim+1);
[id,count] = fscanf(fp,'%d,',1);
while (count == 1)
    samples = samples + 1;
    type = fscanf(fp,'%1s',1);
    if type=='B'
        type =0;
    elseif type=='M'
        type =1;
    else
        type=-1;
        fprintf(' invalid type found in data file %s\n', datafile);
    end
    vec = fscanf(fp,',%e',dataDim);
    train = [train; type vec'];
    [id,count] = fscanf(fp,'%d,',1);
end
if (samples < 569)
    error('Not enough samples');
end

% reorder the rows of "train", if requested
if ~(reord==0)
    p = randperm(samples);
    train(p,:) = train;
end

ntest = round(fracTest * samples);
if ntest < 1
    ntest=1;
elseif ntest >= samples
    ntest = samples-1;
end
ntrain = samples - ntest;

% test = train(1:ntest,:);
% train = train(ntest+1:samples,:);
% modified 4/7/03
test  = train(ntrain+1:samples,:);
train = train(1:ntrain,:);

%--------------------------------------------------------------------------
%
function [ x, y, s ] = ipm_second_order ( A, b, c, Q, TOL, orden );
%
%
% ... una método de puntos interiores que usa el método de Mehrotra
%
%     jl morales 
%     ITAM
%     2013
%
%--------------------------------------------------------------------------

%
% ... calculamos el punto inicial
%
[ x, y, s ] = initial_point( A, b, c, Q);

% Definimos dimensiones y matrices, de tal manera que están ajustadas al
% modelo teórico
n   = length(x); m = length(b);  F = zeros(2*n+m,1);
tau = 0.9995d0;  e = ones(n,1);  
S   = diag(s);   X = diag(x);
sigma = 0.2;

% Definimos las condiciones de F
r_D = Q*x + c - A'*y - s; r_D_norm  = norm(r_D)/(1 + norm(c));
r_P = A*x - b;            r_P_norm  = norm(r_P)/(1 + norm(b));
r_3 = X*s;

% Definimos la función objetivo y la mu y brecha inicial
OBJ =  c'*x + 0.5*x'*Q*x;
mu  = x'*s/n; d_gap = mu;

F   = [ -r_D - diag(1./x)*r_3 ; -r_P  ];  F_norm = norm(F);   iter = 0; 

fprintf('\n');
fprintf('iter   fact_D    fact_P     d_gap         OBJ      \n');
fprintf('---------------------------------------------------\n');

% Establecemos como condición de paro que la brecha del dual y el primal
% sea menor que la tolerancia y limitamos el método a 20 iteraciones
while d_gap > TOL & iter < 20
    
    iter = iter + 1;
    
    KKT = [    Q + diag(1./x)*S       -A'        ;
                     A             zeros(m,m)   ];
    [ L, U ] = lu(KKT);
    %
    % ... obtenemos los pasos del método de Newton
    %
    dw = U\(L\F);
    
    dx =   dw(1:n); 
    ds = - diag(1./x)*( r_3 + S*dx);
    
    alpha_x = step ( x, dx, 1 );  alpha_s = step ( s, ds, 1 );
    %
    % ... calculamos el parámetro de centrado sigma
    %
    
    if (orden==2)
        mu_aff = (x + alpha_x*dx)'*( s + alpha_s*ds)/n;
        sigma  = (mu_aff/mu)^3;
    end
    %
    r_3 = X*s + dx.*ds - sigma*mu*e;
    F  = [ -r_D - diag(1./x)*r_3; -r_P  ]; 
    %
    % ... imprimimos los resultados de la iteración anterior
    %
    fprintf('%3i   %8.2e  %8.2e  %8.2e  % 14.8e \n', iter-1, r_D_norm, r_P_norm, d_gap, OBJ );   
    
    % Calculamos el paso corrector
    
    dw = U\(L\F);
    
    dx =   dw(1:n);  
    dy =   dw(n+1:n+m); 
    ds = - diag(1./x)*( r_3 + S*dx);
    
    alpha_x = step ( x, dx, tau );  
    alpha_s = step ( s, ds, tau );
    alpha = min(alpha_x, alpha_s);
    %
    % ... movemos las variables según el porcentaje calculado
    %
    x = x + alpha_x*dx;  X = diag(x); 
    y = y + alpha_s*dy;    
    s = s + alpha_s*ds;  S = diag(s);
    %
    % ... recalculamos los valores
    %
    OBJ = c'*x + 0.5*x'*Q*x;
    r_D = Q*x + c - A'*y - s; r_D_norm  = norm(r_D)/(1+norm(c));
    r_P = b - A*x;            r_P_norm  = norm(r_P)/(1+norm(b));
    mu  = x'*s/n;          
    d_gap  = mu;
    
    r_3 = X*s;
    F   = [ -r_D - diag(1./x)*r_3 ; -r_P  ];  F_norm = norm(F);
end

fprintf('%3i   %8.2e  %8.2e  %8.2e  % 14.8e \n', iter, r_D_norm, r_P_norm, d_gap, OBJ ); 
%
%--------------------------------------------------------------------------
%
function alpha = step ( x, dx, tau );
one = 1.0d0; zero = 0.0d0;
n = length(x);
alpha  = one;
for i=1:n
    if dx(i) < zero 
        alpha  = min( alpha, -tau*x(i)/dx(i) );
    end 
end 
%
%--------------------------------------------------------------------------
%
function [ x, y, s ] = initial_point( A, b, c, Q );

[m, n] = size(A);

% Usamos la estimación de mínimos cuadrados para obtener la x y y iniciales
mult = A*A';
aux = mult\b;

x = A'*aux ;
y = mult\(A*c);

% Despejamos s, dando por hecho que la primer condición es igual a 0
s = Q*x + c - A'*y;
e = ones(n,1);

% Volvemos positivas a todas las x's y s's
delta_x = max(-3/2*min(x),0);
delta_s = max(-3/2*min(s),0);

x  = x+delta_x*e;
s  = s+delta_s*e;

% Se hace un ajuste heurístico
ro_x = 0.5*(x'*s)/(e'*s);
ro_s = 0.5*(x'*s)/(e'*x);

% Se devuelven x y s calculadas
x  = x+ro_x*e;
s  = s+ro_s*e;
%
%--------------------------------------------------------------------------
%
% ... estandarizar una matriz
%
function [ X ] = scale(X);
[ m, n ] = size(X); e = ones(m,1);

for i=1:n
   xm = mean( X(:,i) );
   xs = std( X(:,i) );
   X(:,i) = (X(:,i) - xm*e)/xs;
end
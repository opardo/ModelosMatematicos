function  ipm_mehrotra();
%--------------------------------------------------------------------------
%... Una prueba del código de puntos interiores extendido a la resolución
% de problemas de programación cuadrática 
%
%                                       JL Morales 2013
%                                       ITAM
%--------------------------------------------------------------------------

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

y = X(1,:);
Y = diag(y);
X(1,:) = [];

A = X*Y; b=y; 

%m = 0; k = 0;

% Obtenemos el número de tumores total y el número de variables del módelo
%mm = m + k;
%nn = 2*n_atr + 2 + m + k + m + k;

fprintf(' Number of samples     ..........  %3i  \n', n_row);
fprintf(' Number of atributes   ..........  %3i  \n', n_atr);
fprintf(' Number of malignant samples ....  %3i  \n', m);
fprintf(' Number of benign samples .......  %3i  \n', k);
fprintf(' Number of variables      .......  %3i  \n', nn);
fprintf(' Number of constraints    .......  %3i  \n', mm);
fprintf(' Penalty parameter  .............  %8.2e  \n', mu);

% Le aplicamos puntos interiores con sigma dinámico
%[ x, lambda, s ] = ipm_method ( A, b, gamma, TOL, orden=2 );

% Le aplicamos puntos interiores con sigma estático
%[ x, lambda, s ] = ipm_method ( A, b, gamma, TOL, orden=1 );

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
function [ x, y, s ] = ipm_method ( A, b, gamma, TOL, orden );
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
% [ x, y, s ] = initial_point( A, b, c, Q);

% Definimos dimensiones y matrices, de tal manera que están ajustadas al
% modelo teórico
[n, m] = size(A); 
e = ones(m,1);
x = (gamma/2)*e; 
s = x;
y = A*x;
X = diag(x);
X_1 = diag(1./x);
S = diag(s);
S_1 = diag(1./s);
lambda = 1;


mu = (x'*z+s'*w)/m;
z = mu*(1./x);
Z = diag(z);
w = mu*(1./s);
W = diag(w);
mu = (x'*z+s'*w)/m;

F = zeros(n+m+1);
tau = 0.9995d0; sigma = 0.2;

% Definimos las condiciones de F
F1 = -e-z-lambda*b+A'*y+w;
F2 = b'*x;
F3 = A*x-y;
F4 = x-gamma*e+s;
F5 = X*z-mu*e;
F6 = S*w-mu*e;

% Definimos la función objetivo y la mu y brecha inicial
OBJ =  (0.5)*y'*y-e'*x;
d_gap = mu;

F   = -[ F1+X_1*F5-S_1*F6+S_1*W*F4; F3; -F2 ];  F_norm = norm(F);   iter = 0; 

fprintf('\n');
fprintf('iter   fact_D    fact_P     d_gap         OBJ      \n');
fprintf('---------------------------------------------------\n');

% Establecemos como condición de paro que la brecha del dual y el primal
% sea menor que la tolerancia y limitamos el método a 20 iteraciones
while d_gap > TOL & iter < 20
    
    iter = iter + 1;
    
    KKT = [   X_1*Z+S_1*W    A'         -b      ;
                   A      -eye(n)     zeros(n,0);  
                  -b'     zeros(m,1)     0     ];
              
    KKT = sparse(KKT);
    [ LL, DD, PP, Sc, neg, ran ] = ldl(KKT);
    %
    % ... obtenemos los pasos del método de Newton
    %
    dt = backsolve( LL, DD, PP, Sc, F );
    
    dx =   dt(1:m);
    dy =   dt(m+1:m+n);
    dlambda = dt(m+n+1);
    
    ds = -F4-dx;
    dz = -X_1(F5+Z*dx);
    dw = -S_1(F6+W*ds);
    
    alpha_x = step_d ( x, dx, 1 );
    alpha_y = step_d ( y, dy, 1 );
    alpha_lambda = step_d ( lambda, dlambda, 1 );
    alpha_s = step_d ( s, ds, 1 );
    alpha_z = step_d ( z, dz, 1 );
    alpha_w = step_d ( w, dw, 1 );
    %
    % ... calculamos el parámetro de centrado sigma
    %
    
    if (orden==2)
        mu_aff = ((x + alpha_x*dx)'*(z + alpha_z*dz)+...
        (s + alpha_s*ds)'*(w + alpha_w*dw)) / m;
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
    
    dt = U\(L\F);
    
    dx =   dt(1:m);
    dy =   dt(m+1:m+n);
    dlambda = dt(m+n+1);
    
    ds = -F4-dx;
    dz = -X_1(F5+Z*dx);
    dw = -S_1(F6+W*ds);
    
    alpha_x = step_d ( x, dx, tau );
    alpha_y = step_d ( y, dy, tau );
    alpha_lambda = step_d ( lambda, dlambda, tau );
    alpha_s = step_d ( s, ds, tau );
    alpha_z = step_d ( z, dz, tau );
    alpha_w = step_d ( w, dw, tau );
    alpha = min(alpha_x, alpha_y, alpha_lambda, alpha_s, alpha_z, alpha_w);
    
    %
    % ... movemos las variables según el porcentaje calculado
    %
    x = x + alpha_x*dx;  
    y = y + alpha_s*dy;
    lambda = lambda + alpha_lambda*dlambda;
    s = s + alpha_s*ds;
    z = z + alpha_z*dz;
    w = w + alpha_w*dw;
    %
    % ... recalculamos los valores
    %
    X = diag(x);
    X_1 = diag(1./x);
    S = diag(s);
    S_1 = diag(1./s);
    lambda = 1;


    mu = (x'*z+s'*w)/m;
    Z = diag(z);
    W = diag(w);       
    d_gap  = mu;
    
    % Redefinimos las condiciones de F
    F1 = -e-z-lambda*b+A'*y+w;
    F2 = b'*x;
    F3 = A*x-y;
    F4 = x-gamma*e+s;
    F5 = X*z-mu*e;
    F6 = S*w-mu*e;

    F   = -[ F1+X_1*F5-S_1*F6+S_1*W*F4; F3; -F2 ]; F_norm = norm(F);

end

fprintf('%3i   %8.2e  %8.2e  %8.2e  % 14.8e \n', iter, r_D_norm, r_P_norm, d_gap, OBJ ); 
%
%--------------------------------------------------------------------------
%
function alpha = step_d ( x, dx, tau );
one = 1.0d0; zero = 0.0d0;
n = length(x);

ind = find( dx<0 );
alpha = min(-x(ind)./dx(ind));
alpha = tau*min (one, alpha);
%
%--------------------------------------------------------------------------
% ... estandarizar una matriz
%
function [ X ] = scale(X);
[ m, n ] = size(X); e = ones(m,1);

for i=1:n
   xm = mean( X(:,i) );
   xs = std( X(:,i) );
   X(:,i) = (X(:,i) - xm*e)/xs;
end


function [ d ] = backsolve ( L, D, P, S, F );

d = P'*(S*F);
d = L\d;
d = D\d;
d = L'\d;
d = P'\d;
d = S*d;
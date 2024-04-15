
function [Best_pos,Best_score,curve]=ISSA(pop,Max_iter,lb,ub,dim,fobj)

ST = 0.6;%Ԥ��ֵ
PD = 0.7;%�����ߵı��У�ʣ�µ��Ǽ�����
SD = 0.3;%��ʶ����Σ����ȸ�ı���

alpha = 0.8; % �������� 0.3-0.8
b = 0.4; % ������״��̬��  0.1-0.5

PDNumber = round(pop*PD); %����������
SDNumber = round(SD*pop);%��ʶ����Σ����ȸ����
if(max(size(ub)) == 1)
    ub = ub.*ones(1,dim);
    lb = lb.*ones(1,dim);
end

%% 1
X0=initializationcircle(pop,dim,ub,lb);
X = X0;

%%���ɷ�����岢ѡ��
X_reverse = repmat(lb, pop, 1) + rand(pop, dim) .* (repmat(ub, pop, 1) - X);
%�����ʼ��Ӧ��ֵ
fitness = zeros(1, pop);
fitness_reverse = zeros(1, pop);

% fitness = zeros(1,pop);
for i = 1:pop
    fitness(i) =  fobj(X(i,:));
    fitness_reverse(i) = fobj(X_reverse(i,:));
end

%% ѡ����Ӧ�ȸ��õĸ���
for i = 1:pop
    if fitness_reverse(i) < fitness(i)
        X(i,:) = X_reverse(i,:);
    end
end

[fitness, index]= sort(fitness);%����
BestF = fitness(1);
WorstF = fitness(end);
GBestF = fitness(1);%ȫ��������Ӧ��ֵ
for i = 1:pop
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%ȫ������λ��
X_new = X;
for i = 1: Max_iter
    %% 2
    ST = 0.6;%Ԥ��ֵ
    b = 0.7;%����ϵ��
    k = 0.1;%�Ŷ�����
    r = b*(tan(-pi*i/(4*Max_iter) + pi/4))-k*rand();
    %��ֹrΪ�������ߴ���1����
    if r<0
        r=0.1;
    end
    if r>1
        r=1;
    end
    PDNumber = round(pop*r);
    SDNumber = round(pop*(1-r));


    BestF = fitness(1);
    WorstF = fitness(end);


    R2 = rand(1);
    for j = 1:PDNumber % ��ÿ��������
        if(R2<ST)
            %% 3
            D = exp(5*cos(pi*(1-(i/Max_iter)^2)));
            v = D * exp(b * rand()) * cos(2*pi*rand());
            Q = normrnd(0,1,[1,dim]); % ������̬�ֲ������
            L = ones(1,dim);
            X(j,:) = v * X(j,:) .* exp(-i / (alpha * Max_iter)) + (R2 >= ST) * (X(j,:) + Q .* L);
        else
            % �������
            X(j,:) = X(j,:) + normrnd(0,1,[1,dim]);
        end
    end
    for j = PDNumber+1:pop
        %        if(j>(pop/2))
        if(j>(pop - PDNumber)/2 + PDNumber)
            X_new(j,:)= randn(1,dim).*exp((X(end,:) - X(j,:))/j^2);
        else
            %����-1��1�������
            A = ones(1,dim);
            for a = 1:dim
                if(rand()>0.5)
                    A(a) = -1;
                end
            end
            AA = A'*inv(A*A');
            X_new(j,:)= X(1,:) + abs(X(j,:) - X(1,:)).*AA';
        end
    end
    Temp = randperm(pop);
    SDchooseIndex = Temp(1:SDNumber);
    for j = 1:SDNumber
        if(fitness(SDchooseIndex(j))>BestF)
            X_new(SDchooseIndex(j),:) = X(1,:) + randn().*abs(X(SDchooseIndex(j),:) - X(1,:));
        elseif(fitness(SDchooseIndex(j))== BestF)
            K = 2*rand() -1;
            X_new(SDchooseIndex(j),:) = X(SDchooseIndex(j),:) + K.*(abs( X(SDchooseIndex(j),:) - X(end,:))./(fitness(SDchooseIndex(j)) - fitness(end) + 10^-8));
        end
    end
    %�߽����
    for j = 1:pop
        for a = 1: dim
            if(X_new(j,a)>ub(a))
                X_new(j,a) =ub(a);
            end
            if(X_new(j,a)<lb(a))
                X_new(j,a) =lb(a);
            end
        end
    end
    %����λ��
    for j=1:pop
        fitness_new(j) = fobj(X_new(j,:));
    end
    for j = 1:pop
        if(fitness_new(j) < GBestF)
            GBestF = fitness_new(j);
            GBestX = X_new(j,:);
        end
    end
    %% 4
    Avgf = mean(fitness_new);
    x0 = rand;
    for j = 1:pop
        if fitness_new(j)<Avgf %��������
            Temp = X_new(j,:);
            r = tan((rand() - 0.5)*pi);%���������
            Temp = Temp + r.*Temp;
            Temp(Temp>ub) = ub(Temp>ub);
            Temp(Temp<lb) = lb(Temp<lb);
            fitTemp = fobj(Temp);
            if fitTemp<fitness_new(j)
                fitness_new(j) = fitTemp;
                X_new(j,:) = Temp;
            end
        else%Tent�Ŷ�
            if x0<0.5
                tentV = 2*x0+rand/pop;
            else
                tentV = 2*(1-x0)+rand/pop;
            end
            Temp = X_new(j,:);
            Temp = Temp + tentV.*Temp;
            Temp(Temp>ub) = ub(Temp>ub);
            Temp(Temp<lb) = lb(Temp<lb);
            fitTemp = fobj(Temp);
            if fitTemp<fitness_new(j)
                fitness_new(j) = fitTemp;
                X_new(j,:) = Temp;
            end
            x0 = tentV;
        end
    end

    X = X_new;
    fitness = fitness_new;
    %�������
    [fitness, index]= sort(fitness);%����
    BestF = fitness(1);
    WorstF = fitness(end);
    for j = 1:pop
        X(j,:) = X(index(j),:);
    end
    curve(i) = GBestF;
end
Best_pos =GBestX;
Best_score = curve(end);
end




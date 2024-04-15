%% cat ӳ��
function Xout = icircleMap(dim)
    a = 3.85;
    b = 0.4;
    c = 0.7 / (3.85 * pi);
    %��ʼ������
    x1 = zeros(1,dim);
    y1 = zeros(1,dim);
    x = rand(dim);
    y = rand(dim);
    N = 1;
    for i = 1:dim
        x1(i) = mode(a * x(i) + b - c * sin(a * pi * x(i)), N);
        % y1(i) = mode(a*x(i) + a.*b.*y(i),N);
    end
    Xout = (x1 - min(x1))./(max(x1) - min(x1));
end
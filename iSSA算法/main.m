%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ���������п�ʼ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% �����ʼ�׶�
clear all 
close all
clc

%% ��ʼ�����㷨����
SearchAgents_no=30; %��Ⱥ������С
N=SearchAgents_no;
pop_num=SearchAgents_no;
M=5;
Function_name='F7'; %ѡ����Ժ���
Max_iteration=300; %����������
Max_iter=Max_iteration;

%% ��ʼ���㷨�߽�
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); 
 X = initialization_3(N, dim, ub, lb);

%% ���������Ż��㷨 
[Best_pos,Best_score,SSA_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);                         %1.ԭʼ��ȸ�㷨    ��SSA)
[Best_pos_ILL_SSA,Best_score_ILL_SSA,ILL_SSA_curve]=ISSA_2036(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); %2.�Ľ���ȸ�㷨    (ISSA)
[PSO_gBestScore,PSO_gBest,PSO_cg_curve]=PSO(N,Max_iteration,lb,ub,dim,fobj);                               %3.ԭʼ����Ⱥ�㷨  ��PSO)
[fMin_GWO,bestX_GWO,GWO_curve]=GWO(pop_num,Max_iter,lb,ub,dim,fobj);                                       %4.ԭʼ�����Ż��㷨 (GWO) 
[fMin_WOA,bestX_WOA,WOA_curve]=WOA(pop_num,Max_iter,lb,ub,dim,fobj);                                       %5.ԭʼ�����Ż��㷨��WOA) 
% [ABest_scoreChimp,ABest_posChimp,Chimp_curve]=Chimp(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);         %6.ԭʼ�������㷨  (Chimp)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ���滻�ĸ����Ż��㷨 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[fMin_EWOA,bestX_EWOA,EWOA_curve]=E_WOA(pop_num,Max_iter,lb,ub,dim,fobj);%�Ľ������Ż��㷨��EWOA) 
%[Best_pos_ISSA,Best_score_ISSA,ISSA_curve]=ISSA(M,Max_iteration,lb,ub,dim,fobj);%�Ľ���ȸ�㷨1(ISSA_1)
%[PSO_gBestScore,PSO_gBest,PSO_cg_curve]=MPSO(N,Max_iteration,lb,ub,dim,fobj);%�Ľ�����Ⱥ�㷨��MPSO)
%[PSO_gBestScore,PSO_gBest,PSO_cg_curve]=TACPSO(N,Max_iteration,lb,ub,dim,fobj);%����ʱ��Ľ�������Ⱥ�㷨��TACPSO)
%[fMin_IGWO,bestX_IGWO,IGWO_curve]=IGWO(pop_num,Max_iter,lb,ub,dim,fobj);%�Ľ������Ż��㷨(IGWO)
%[HHO_gBestScore,HHO_gBest,HHO_cg_curve]=HHO(N,Max_iteration,lb,ub,dim,fobj);%ԭʼ����˹ӥ�㷨��HHO)
%[ESCA_Best_score, ESCA_Best_pos, ESCA_Curve] = ESCA(X, N, Max_iteration, lb, ub, dim, fobj);%�Ľ��������㷨��ESCA)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ���Ʋ��Ժ���ͼ��
figure
func_plot(Function_name);
title('�����ռ�')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])

%% ���Ƶ�������ͼ��

figure;
t = 1:Max_iteration;
% semilogy(t, SSA_curve, 'blacko-',t,  ILL_SSA_curve, 'ro-',t, PSO_cg_curve, 'yo-',t,GWO_curve,'bO-',t,WOA_curve,'mo-', 'linewidth', 1.5, 'MarkerSize', 6, 'MarkerIndices', 1:25:Max_iteration);
% ����t, SSA_curve�ȶ��Ѿ�����
hLines = semilogy( t, SSA_curve, 'o-',t, ILL_SSA_curve, 'o-', t, PSO_cg_curve, 'o-', t, GWO_curve, 'o-', t, WOA_curve, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
set(hLines, 'MarkerIndices', 1:25:Max_iteration); % ����Max_iteration�Ѷ���

% ����ÿ���ߵ���ɫ
% C1 = addcolor(1);
% C2 = addcolor(98);
% C3 = addcolor(135);
% C4 = addcolor(214);
% C5 = addcolor(215);

% ����ʹ��set��������ÿ���ߵ���ʽ
set(hLines(1), 'LineStyle', '--', 'Marker', 'd', 'LineWidth', 1, 'Color', [252,140,90]/255);
set(hLines(2), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 1, 'Color', [219,49,36]/255);
set(hLines(3), 'LineStyle', '--', 'Marker', '^', 'LineWidth', 1, 'Color', [144,190,224]/255);
set(hLines(4), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 1, 'Color', [75,116,178]/255);
set(hLines(5), 'LineStyle', '--', 'Marker', 'd', 'LineWidth', 1, 'Color',[255,223,146]/255);



% title(Function_name)
xlabel('��������');
ylabel('��Ӧ��ֵ');
axis fill
% grid on                                                        
box on
legend('SSA','ISSA','PSO','GWO','WOA');
set(gca,'fontname','���� ')
% ylim([-12569.5, 0]);

%% ������㷨���Ž��
display(['SSA����ֵ : ', num2str(Best_score)]);
display(['ILL_SSA����ֵ : ', num2str(Best_score_ILL_SSA)]);
display(['PSO����ֵ : ', num2str(PSO_gBestScore)]);
% display(['Chimp����ֵ : ', num2str(ABest_scoreChimp)]);
display(['GWO����ֵ : ', num2str(fMin_GWO)]);
display(['WOA����ֵ : ', num2str(fMin_WOA)]);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ���������н��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
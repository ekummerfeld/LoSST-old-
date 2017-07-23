%%
%x is data, schedule is list of times to produce graphs at, and if empty
%will use probabilistic method instead.  e is the cutoff for PC to use, and
%batch is a boolean value: true will prevent the function from producing
%lots of extra output, false will permit the function to do so.
%plotgraphs is a boolean value: true will permit the function to make
%vidual plots of every output graph after PC is run.  This should be left
%false unless you're certain that the number of output graphs is small,
%otherwise MATLAB could lock up and become unresponsive.
%some other parameters are currently built into the code itself rather than
%used as inputs to the OSLM() function, especially the burnin parameters,
%the reweighting parameter ratpara, and the threshold for reweighting
%scpara
function output = OSLM(x,schedule,e,batch,plotgraphs)
%%
%initialize
addpath('BNT')
addpath(genpathKPM(pwd))

datasize=size(x);
mu=mean(x(1:10,:));  %zeros(1,datasize(2));

%mu=mean(x);

%d=zeros(1,datasize(2));
tcov=cov(x(1:10,:));
cor=zeros(datasize(2));

%tcov=cov(x);

output.graphtimes=[];
output.graphs{1}=[];

a=ones(1,datasize(1));
b=ones(1,datasize(1));
b(1)=2;
sample_size=ones(1,datasize(1));
%sensitivity for graph search
%e=.05;

tcovstore={};
pdag_count=0;

M_error=ones(1,datasize(1))*datasize(2);
pverr=ones(1,datasize(1));
priorss=datasize(2)+5; %prior sample size
priorerrval=.2; %prior error value
experrvalpr=.1;
experrval=ones(1,datasize(1))*experrvalpr; %expected error value (unbiased estimated based on observations)
%current sample size is also used, but is initialized separately

trigger=0;

%for LRE
initstbias=0;  %this is something like an initial stability bias
pval=ones(1,datasize(1))*.5;
ntrack=zeros(1,datasize(1));
Q=ones(1,datasize(1))*initstbias;
sumsqrw=ones(1,datasize(1));
poolp=zeros(1,datasize(1));

burnin=10;%burnin=datasize(2)*1.05;  %this determines the length of the burn-in period %!_!_!_using parfindFOUR
burnin_MD=chi2inv(.5,datasize(2));  %this is the Mahalanobis Distance to use during the burn-in period

plearn=zeros(1,datasize(1));
make_graph=0;

fol=.005; %frequency of learning parameter, for probabilistic scheduler.

%scale and lower bound parameter for transforming poolp values to weights
scpara=.95;   %normal parameter: .95
ratpar=3;  %!_!_!_using parfindFOUR  %normal parameter: 1.5
%parameter for ratio-type downweighting.  as ratpar ->1, curve steapens/downweights more heavily.
%also determines maximum downweight ratio, equal to 1/ratpar (i.e.,
%ratpar=1 downweights to an effective sample size of 0 at poolp(j)=1,
%ratpar=2 cuts effective sample size in half at poolp(j)=1
%%
for j=1:datasize(1)
%%
    %calc accumulating error rate of correlation
    %use Mahalanobis error to calc error of new point from old tcov and mu
    %prob want regular M_error here, not normed error.  Take account for
    %datasize(2) in the distributional part.
    
    %M_error(j) = (x(j,:)-mu)*inv(tcov)*(x(j,:)-mu)';
    
    %__trying it a new way___
    M_error(j) = (x(j,:)-mu)/tcov*(x(j,:)-mu)';

    norm_M_error(j)=M_error(j)/datasize(2);
    
    %%
        %Update tcovariance Matrix
    %use learning rate to update tcov
    
    %replaces commented out update of b(j+1) below
    if j>1
        b(j)=b(j-1)+a(j);
    end
    
    %regular OCME
    d=(a(j)/b(j))*(x(j,:)-mu);
    mu=mu+d;
    for i=1:datasize(2)
        for k=1:datasize(2)
            tcov(i,k)=(1/b(j))*((b(j)-a(j))*tcov(i,k)+(b(j)-a(j))*d(i)*d(k)+a(j)*(x(j,i)-mu(i))*(x(j,k)-mu(k)));
        end
    end

    tcovstore{j}=tcov;
    %%
    %update learning rate
    %need to track the weighted sum of M_error values, and compare this
    %against a distribution which depends on: sample size, datasize(2)
    
    if j>1
        sample_size(j)=(a(j-1)/a(j))*sample_size(j-1)+1;
    else
        sample_size(j)=1;
    end
    
    %not sample size, actually.  need to track sum of _squared_ weights
    %directly.
    
    %P = normcdf(X,mu,sigma)
    %P = chi2cdf(X,V)
    %X = norminv(P,mu,sigma)
    %calc...  norminv(chi2cdf(M_error(j),datasize(2)),0,1)
    %if j>1
    %    %the min is to prevent ntrack(j)=Inf, which causes HUGE PROBLEMS
    %    ntrack(j)=norminv(min(chi2cdf(M_error(j),datasize(2)),.999),0,1);
    %    Q(j)=Q(j-1)+a(j)*ntrack(j);
    %    sumsqrw(j)=sumsqrw(j-1)+a(j)^2;
    %    poolp(j)=normcdf(Q(j),0,sqrt(sumsqrw(j)));
    %end
    
    %during the burn-in period:
    if j>1&&sample_size(j-1)<=burnin
        %gotta make sure the right things get burned in
        pval(j)=.5;
        %the min is to prevent ntrack(j)=Inf, which causes HUGE PROBLEMS
        ntrack(j)=norminv(min(pval(j),.999),0,1);
        Q(j)=Q(j-1)+a(j)*ntrack(j);
        sumsqrw(j)=sumsqrw(j-1)+a(j)^2; %sum of squared weights
        poolp(j)=normcdf(Q(j),0,sqrt(sumsqrw(j)));
        a(j+1)=a(j);
        if trigger==1
            experrval(j)=(experrval(j-1)*(sample_size(j)-1)+chi2cdf(M_error(j),datasize(2))-.5)/(sample_size(j));
        else
            experrval(j)=experrvalpr;
        end
        
        pcheck=1;
        trigger=1;
    end
    
    %after the burn-in period is over:
    if j>1&&isnan(M_error(j))==0&&M_error(j)>=0&&sample_size(j-1)>burnin  %j>burnin for the burn-in period
        trigger=0;
        %Calculating pooled p values and turning them into weights
        if poolp(j-1)<=0
            pcheck=0;
        end
        
        %experrval(j)=(experrval(j-1)*(sample_size(j)-1)+chi2cdf(M_error(j),datasize(2))-(.5+pverr(j-1)))/(sample_size(j));
        %pverr(j)=max(min((priorss*priorerrval+sample_size(j)*experrval(j))/(priorss + sample_size(j)),pverr(j-1)),0);
        
        pval(j)=fcdf((sample_size(j)-datasize(2))/(datasize(2)*(sample_size(j)-1))*M_error(j),datasize(2),sample_size(j)-datasize(2));
        
        %pval(j)=max(chi2cdf(M_error(j),datasize(2))-min(pverr(j),pcheck),0);   %***Rewrite this in terms of F-distribution/hotelling's t-square***
        %pval(j)=chi2cdf(M_error(j)-.03*12*1/sqrt(sample_size(j-1))*20,datasize(2));
        %the min/maxes are to prevent ntrack(j)=Inf, which causes HUGE PROBLEMS
        ntrack(j)=norminv(max(min(pval(j),.9999),.0001),0,1); %inverse normal cdf of the pvalue
        Q(j)=Q(j-1)+a(j)*ntrack(j); %weighted sum of inverse normal cdf of p-values
        sumsqrw(j)=sumsqrw(j-1)+a(j)^2; %sum of squared weights
        poolp(j)=normcdf(Q(j),0,sqrt(sumsqrw(j))); %pooled pvalue is the appropriate normal cdf of Q(j)
        
%         plearn(j)=plearn(j-1)+fol*poolp(j);%fol is frequency param, square for scaling
        plearn(j)=plearn(j-1)+fol*(poolp(j)-poolp(j)*plearn(j-1));%fol is frequency param
        
        %this needs to be squashed so that a regular-ish p-value (.5)
        %doesn't cause massive downweighting.
        if poolp(j)<scpara
            a(j+1)=a(j);
        else            
            a(j+1)=1/(1-1/ratpar*((1/(1-scpara))*poolp(j)-(1/(1-scpara)-1)))*a(j);
            
            %previous versions:
            %the min is to prevent sample sizes from getting below a
            %certain value, since bad stuff happens if it does
            %a(j+1)=max((expinv((1/(1-scpara))*poolp(j)-(1/(1-scpara)-1))+1)*a(j),a(j));
            
            %a(j+1)=min(b(j)*.02,max((expinv((1/(1-scpara))*poolp(j)-(1/(1-scpara)-1))^(.5)+1)*a(j),a(j)));
        end
        %a(j+1)=max(expinv(poolp(j)^2)*b(j),a(j));  %expinv is for rescaling the poolp.
        
        %____threshold method____
        %if poolp(j)<.99
        %    a(j+1)=a(j);
        %else
        %    a(j+1)=b(j)*.1;
        %end
    end
    %%
    %learn PDAG matrix from the correlation matrices

    
    if isempty(schedule)
        %probabilistic scheduler
        if rand(1)<plearn(j) && j>24
            make_graph=1;
            %delta_alpha_MTDL=0;
            plearn(j)=0;
        end
    else
        %Do full run for particular time steps
        if ismember(j,schedule)==1
            make_graph=1;
        else
            make_graph=0;
        end
    end
    %PC search for graph, then plot it
    if make_graph==1
        %%
        %calc correlations
        for i=1:datasize(2)
            for k=1:datasize(2)
                cor(i,k)=tcov(i,k)/sqrt(tcov(i,i)*tcov(k,k));
            end
        end
        pdag_count=pdag_count+1; %index the pdags amongst themselves
        %pdag_index(pdag_count)=j; %index the pdags amongst the timesteps
        %use the bayes net toolbox to calculate the pdag matrix
        %pdag{pdag_count} = learn_struct_pdag_pc('cond_indep_fisher_z', length(cor), length(cor), cor, floor(sample_size(j)), e);
        %plot pdag
        output.graphtimes(pdag_count)=j;
        %yes=floor(sample_size(j))
        
        %___uncomment the below eventually!!!!!!!!!!!!!!!!!!!!!____
        %cor
        output.graphs{pdag_count}=learn_struct_pdag_pc('cond_indep_fisher_z', length(cor), length(cor), cor, floor(sample_size(j)), e);
        %output.graphs{pdag_count}=ones(10);
        
        if plotgraphs
            figure('NextPlot','new')
            draw_graph(abs(output.graphs{pdag_count}))
            title(j)
        end
        make_graph=0;
    end
end
%%
%comparing poolp w/ batch equivalent weighting to the known analytic
%solution: chi-square distribution with DOF=#data*variables
x=0;
chisquaretest=zeros(1,datasize(1));
for j=1:length(M_error)
    if j<burnin
        x=x+burnin_MD;
    else
        x=x+M_error(j);
    end
    chisquaretest(j)=chi2cdf(x,j*datasize(2));
end

%%
if ~batch
    plot(sample_size)
    title('sample size')
    figure('NextPlot','new')
    plot(norm_M_error,'red')
    %hold on
    %plot(MTDL)
    %hold off
    figure('NextPlot','new')
    plot(poolp)
    hold on
    plot(chisquaretest,'red')
    title('Comparing pooled p-value to batch analytic solution')
    legend('pooled p-value','analytic p-value')
    hold off
    %figure('NextPlot','new')
    %plot(pval)
    %title pval
    output.sumsqrw=sumsqrw;
    output.poolp=poolp;
    output.chisquaretest=chisquaretest;
    output.Q=Q;
    output.a=a;
    output.M_error=M_error;
    output.ntrack=ntrack;
    output.tcovstore=tcovstore;
    output.b=b;
    output.sample_size=sample_size;
    output.pverr=pverr;
    output.experrval=experrval;
    output.plearn=plearn;
    
    output.pval=pval;
    
    if isempty(schedule)
        output.plearn=plearn;
    end
end
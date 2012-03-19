#include <shogun/classifier/NewtonSVM.h>
#include <shogun/features/Labels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CNewtonSVM::CNewtonSVM() : CLinearMachine()
{
}

CNewtonSVM::CNewtonSVM(CDotFeatures* traindat, CLabels* trainlab,bool linear,float64_t lambda,Options opt)
: CLinearMachine()
{
	set_features(traindat);
	set_labels(trainlab);
}

CNewtonSVM::~CNewtonSVM()
{
}
void initparam(int32_t n,int32_t d)
{	x_n=num_vec;
	x_d=num_feat;
	w=SG_MALLOC(float64_t,n);
	sol=SG_MALLOC(float64_t,n+1);
	wlen=n;
	sol_len=n+1;
}
bool CNewtonSVM::train_machine(CFeatures* data)
{
	ASSERT(labels);
	//following if is executed only if data is sent through train_machine()
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(features);
	bool converged=false;
	int32_t iter=0;
	train_labels=labels->get_int_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();
	
	ASSERT(num_vec==train_labels.vlen);
	//SG_FREE(w);
	//w_dim=num_feat;
	//w=SG_MALLOC(float64_t, num_feat);
	//float64_t* output=SG_MALLOC(float64_t, num_vec);
	initparam(num_vec,num_feat)	
	if(linear){
		if(!opt.cg){
			primal_svm_linear(features);
		}else{
			primal_svm_linear_cg(data);
		}
	}
	else{
		if(!opt.cg){
			primal_svm_nonlinear(data);
		}else{
			primal_svm_nonlinear_cg(data);
		}
	}
	b=w[num_vec];

	train_labels.free_vector();

	return converged;
}
void primal_svm_linear(CDotFeatures* feat)
{	
	SG_FREE(w);
	w= SG_MALLOC(float64_t,x_d+1);
	int32_t iter=0;
	float64_t *out=SG_MALLOC(float_64_t,x_n);
	
	while(1)
	{
		iter++;
		if(iter>opt.iter_max_Newton)
		{
			SG_SPRINT("Maximum number of Newton steps reached.Try larger lambda");
		break;
		}
	}
	obj_fun_linear(features,out);
	
	if(opt.lin_cg)
	{	
		












}
/* Compute the objective function, its gradient and the set of support vectors
Out is supposed to contain 1-Y.*(X*w)
*/
void obj_fun_linear(CDotFeatures* feat,float64_t *out)
{
	for(int32_t i=0;i<x_n;i++)
	{
		if(out[i]<0)
		out[i]=0;
	}
	//create copy of w0 	
	float64_t *w0=SG_MALLOC(float64_t,x_n+1)
	memcpy(w0,w,sizeof(float64_t)*(x_n+1));
	w0[x_n]=0;
	//create copy of out
	float64_t *out1=SG_MALLOC(float64_t,x_n)
	memcpy(out1,out,sizeof(float64_t)*(x_n));
	//compute steps
	out1=CMath::vector_multiply(out1,out,out,x_n);
	float64_t p1=CMath::sum(out1)/2;	
	float64_t C;
	CMath::dgemm((lambda/2),w0,x_n,1,CblasTrans,w,1,CblasNoTrans,0.0,C);
	obj=p1+C;
	CMath::scale_vector(lambda,w0,x_n+1);
	float64_t *temp=SG_MALLOC(float64_t,x_dn;
	CMath::vector_multiply(temp,out,train_lab,x_n);
	float64_t *temp1=SG_MALLOC(float64_t,x_n);
		
	CMath::dgemm(1.0,temp,x_n,1,CblasTrans,features,x_d,CblasNoTrans,0.0,temp1);
	float64_t *temp3=SG_MALLOC(float64_t,x_n+1);
	memcpy(temp3,temp1,x_n);
	temp3[x_n]=CMath::sum(temp);
	sv=SG_MALLOC(int32_t,x_n);
	sv_len=0;
	for(int32_t i=0;i<x_n;i++)
	{	
		if(out[i]>0)
		sv[sv_len++]=i;
	}
}
		

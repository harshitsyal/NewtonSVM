/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-Harshit Syal
 * Copyright (C) 2006-2012 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/classifier/svm/NewtonSVM.h>
#include <shogun/features/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

using namespace shogun;

CNewtonSVM::CNewtonSVM()
: CLinearMachine(), lambda(1),epsilon(1e-5), use_bias(true)
{
}

CNewtonSVM::CNewtonSVM(
	float64_t l,int32_t itr, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{	lambda=l;
	num_iter=itr;
	epsilon=1e-5;
	prec=1e-6;
	num_iter=20;
	use_bias=true;
	set_features(traindat);
	set_labels(trainlab);
}


CNewtonSVM::~CNewtonSVM()
{
}


bool CNewtonSVM::train_machine(CFeatures* data)
{	
	ASSERT(labels);

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}

	ASSERT(features);

	SGVector<float64_t> train_labels=labels->get_labels();
	SGMatrix<float64_t> mat=features->get_computed_dot_feature_matrix();	
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();
	
	//Assigning dimensions for whole class scope
	x_n=num_vec;
	x_d=num_feat;	
	
	ASSERT(num_vec==train_labels.vlen);
	
	SG_FREE(w);
	float64_t *w = SG_MALLOC(float64_t,x_d+1);
	memset(w,0,sizeof(float64_t)*(x_d+1));

	float64_t *out=SG_MALLOC(float64_t,x_n);
	for(int32_t i=0;i<x_n;i++)
		out[i]=1;


	int32_t *sv=SG_MALLOC(int32_t,x_n),size_sv,iter=0;
	float64_t obj,*grad=SG_MALLOC(float64_t,x_d+1);
		
	while(1)
	{
		iter++;
		if(iter>num_iter)
		{
			SG_SPRINT("Maximum number of Newton steps reached.Try larger lambda");
		break;
		}
	
		
	obj_fun_linear(w,features,out,&obj,sv,&size_sv,grad,train_labels);
	
	float64_t *Xsv = SG_MALLOC(float64_t,x_d*size_sv);
	for(int32_t k=0;k<size_sv;k++)
	{
		 for (int32_t j=0; j<x_d; j++)
		 	Xsv[k+j*size_sv]=mat.matrix[sv[k]+j*x_n];
		
	
	}
	

	float64_t *lcrossdiag=SG_MALLOC(float64_t,(x_d+1)*(x_d+1)),*vector=SG_MALLOC(float64_t,x_d+1);
	for(int32_t i=0;i<x_d;i++)
	vector[i]=lambda;
	vector[x_d]=0;
	createDiagnolMatrix(lcrossdiag,vector,x_d+1);
	
	float64_t *Xsv2=SG_MALLOC(float64_t,x_d*x_d);
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,x_d,x_d,size_sv,1.0,Xsv,size_sv,Xsv,size_sv,0.0,Xsv2,x_d);		
	float64_t *sum=SG_MALLOC(float64_t,x_d);
	memset(sum,0,sizeof(float64_t)*x_d);
	
	for(int32_t j=0;j<x_d;j++)
	{	
		for(int32_t i=0;i<size_sv;i++)
			sum[j]+=Xsv[i+j*size_sv];
	}
	float64_t *Xsv2sum=SG_MALLOC(float64_t,(x_d+1)*(x_d+1));
	
	for (int32_t i=0; i<x_d; i++)
    	{
        	
        	 for (int32_t j=0; j<x_d; j++)
        		 Xsv2sum[j*(x_d+1)+i]=Xsv2[j*x_d+i];
		Xsv2sum[x_d*(x_d+1)+i]=sum[i];
    	}
	for (int32_t j=0;j<x_d;j++)
	Xsv2sum[j*(x_d+1)+x_d]=sum[j];
	Xsv2sum[x_d*(x_d+1)+x_d]=size_sv;
	
	float64_t *identity_matrix=SG_MALLOC(float64_t,(x_d+1)*(x_d+1));
	
	for(int32_t i=0;i<x_d+1;i++)
	vector[i]=1;
	createDiagnolMatrix(identity_matrix,vector,x_d+1);
	
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,x_d+1,x_d+1,x_d+1,1.0,lcrossdiag,x_d+1,identity_matrix,x_d+1,1.0,Xsv2sum,x_d+1);		
	float64_t *inverse=SG_MALLOC(float64_t,(x_d+1)*(x_d+1));
	int32_t r=x_d+1;
	
	CMath::pinv(Xsv2sum,r,r,inverse);
	
	float64_t *step=SG_MALLOC(float64_t,r),*s2=SG_MALLOC(float64_t,r);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,r,1,r,1.0,inverse,r,grad,r,0.0,s2,r);		
	for(int32_t i=0;i<r;i++)
		step[i]=-s2[i];

	float64_t t;
	
	line_search_linear(w,step,out,train_labels,lambda,&t);
	CMath::vec1_plus_scalar_times_vec2(w,t,step,r);
	float64_t newton_decrement;
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,1,r,-0.5,step,r,grad,r,0.0,&newton_decrement,1);		
	SG_SPRINT("Itr=%d,Obj=%f,No of sv=%d, Newton dec=%0.3f,line search=%0.3f\n\n",iter,obj,size_sv,newton_decrement,t);	
	if(newton_decrement*2<prec*obj)
	break;

}
	
	SG_SPRINT("FINAL W AND BAIS Vector=\n\n");
	CMath::display_matrix(w,x_d+1,1);	
	set_w(SGVector<float64_t>(w, x_d));
	set_bias(w[x_d]);
	return true;


}

 
void CNewtonSVM::line_search_linear(float64_t *w,float64_t *d,float64_t *out,SGVector<float64_t> Y,float64_t lambda,float64_t *tx)
{	
	SGMatrix<float64_t> mat=features->get_computed_dot_feature_matrix();
	float64_t *dcopy=SG_MALLOC(float64_t,x_d+1);
	
	float64_t *outz=SG_MALLOC(float64_t,x_n);

	float64_t *temp1=SG_MALLOC(float64_t,x_n);
	float64_t *outzsv=SG_MALLOC(float64_t,x_n);
	float64_t *Ysv=SG_MALLOC(float64_t,x_n);
	float64_t *Xsv=SG_MALLOC(float64_t,x_n);
	float64_t *temp2=SG_MALLOC(float64_t,x_d);
	for(int32_t i=0;i<x_d+1;i++)
	dcopy[i]=d[i];	
	float64_t t=0.0;
	float64_t *Xd=SG_MALLOC(float64_t,x_n);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,x_n,1,x_d,1.0,mat.matrix,x_n,d,x_d,0.0,Xd,x_n);	
	
	CMath::add_scalar(d[x_d],Xd,x_n);
	float64_t wd;
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,1,x_d,lambda,w,x_d,d,x_d,0.0,&wd,1);		
	

	float64_t tempg,dd;
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,1,x_d,lambda,d,x_d,d,x_d,0.0,&dd,1);	
	float64_t g,h;
	int32_t sv_len=0,*sv=SG_MALLOC(int32_t,x_n);
	

	do
	{
	
		CMath::vector_multiply(temp1,Y.vector,Xd,x_n);
		CMath::scale_vector(t,temp1,x_n);
		CMath::add(outz,1.0,out,-1.0,temp1,x_n);	
	// sv = find(outz>0);
		sv_len=0;
		for(int32_t i=0;i<x_n;i++)
		{	
			if(outz[i]>0)
			sv[sv_len++]=i;
		}
	
	//g = wd + t*dd - (outz(sv).*Y(sv))'*Xd(sv); % The gradient (along the line)
	for(int32_t i=0;i<sv_len;i++)
	{	outzsv[i]=outz[sv[i]];
		Ysv[i]=Y.vector[sv[i]];
		Xsv[i]=Xd[sv[i]];
	}
			
	CMath::vector_multiply(temp1,outzsv,Ysv,sv_len);
	
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,1,sv_len,1.0,temp1,sv_len,Xsv,sv_len,0.0,&tempg,1);	
	g=wd+(t*dd)-tempg;

	// h = dd + Xd(sv)'*Xd(sv); % The second derivative (along the line)
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,1,sv_len,1.0,Xsv,sv_len,Xsv,sv_len,0.0,&h,1);	
	
	h+=dd;
	// t = t - g/h; % Take the 1D Newton step. Note that if d was an exact Newton
	t-=g/h;
	}while(((g*g)/h)>1e-10);

	for(int32_t i=0;i<x_n;i++)
		out[i]=outz[i];
	*tx=t;


	SG_FREE(temp1);
	SG_FREE(temp2);
	SG_FREE(outz);
	SG_FREE(outzsv);
	SG_FREE(Ysv);
	SG_FREE(Xsv);
}
	


void CNewtonSVM::obj_fun_linear(float64_t *w,CDotFeatures *features,float64_t *out,float64_t *obj,int32_t *sv,int32_t *numsv,float64_t *grad,SGVector<float64_t> v)
{	
	
	SGMatrix<float64_t> mat=features->get_computed_dot_feature_matrix();	
	float64_t lambda=get_lambda();
	// out=max(0,out);
	for(int32_t i=0;i<x_n;i++)
	{
		if(out[i]<0)
		out[i]=0;
	}
	
	//create copy of w0
	float64_t *w0=SG_MALLOC(float64_t,x_d+1);
	memcpy(w0,w,sizeof(float64_t)*(x_d));
	w0[x_d]=0;//do not penalize b

	//create copy of out
	float64_t *out1=SG_MALLOC(float64_t,x_n);
	
	//compute steps
	
	//for obj
	
	CMath::vector_multiply(out1,out,out,x_n);
	
	float64_t p1=CMath::sum(out1,x_n)/2;	
	
	float64_t C;
	float64_t *w0copy=SG_MALLOC(float64_t,x_d+1);
	memcpy(w0copy,w0,sizeof(float64_t)*(x_d+1));
	CMath::scale_vector(0.5,w0copy,x_d+1);
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,1,x_d+1,lambda,w0,x_d+1,w0copy,x_d+1,0.0,&C,1);	
	*obj=p1+C;
	//SG_SPRINT("obj=%f",*obj);
	

	//for gradient
	//grad = lambda*w0 - [((out.*Y)'*X)'; sum(out.*Y)];
	
	CMath::scale_vector(lambda,w0,x_d);
	
	float64_t *temp=SG_MALLOC(float64_t,x_n);//temp = out.*Y 
	CMath::vector_multiply(temp,out,v.vector,x_n);
	
	float64_t *temp1=SG_MALLOC(float64_t,x_n);
	
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,1,x_d,x_n,1.0,temp,x_n,mat.matrix,x_n,0.0,temp1,1);		
	int32_t one=1,dim=x_d;	
	CMath::transpose_matrix(temp1,one,dim);
	
	float64_t *p2=SG_MALLOC(float64_t,x_d+1);
	for(int32_t i=0;i<x_d;i++)
	p2[i]=temp1[i];

	p2[x_d]=CMath::sum(temp,x_n);
	CMath::add(grad,1.0,w0,-1.0,p2,x_d+1);
	
	
	int32_t sv_len=0;
	for(int32_t i=0;i<x_n;i++)
	{	
		if(out[i]>0)
		sv[sv_len++]=i;
	}
	*numsv=sv_len;
	SG_FREE(w0);
	SG_FREE(out1);
	SG_FREE(temp);
	SG_FREE(temp1);
	SG_FREE(p2);

}


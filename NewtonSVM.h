/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _NEWTONSVN_H___
#define _NEWTONSVM_H___

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{

class CNewtonSVM : public CLinearMachine
{
	public:
		/** default constructor */
		CNewtonSVM();

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CPerceptron(CDotFeatures* traindat, CLabels* trainlab,bool linear,float64_t lambda,Options opt);
		virtual ~CPerceptron();

		/** get classifier type
		 *
		 * @return classifier type PERCEPTRON
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_NEWTONSVM; }

		/// set maximum number of iterations
		inline void set_max_iter(int32_t i)
		{
			max_iter=i;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "NewtonSVM"; }
		virtual bool train_machine(CFeatures* data=NULL);
	protected:
		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		
		void obj_fun_linear(CDotFeatures* feat,float64_t *out);
		void primal_svm_linear(CDotFeatures* feat)

	protected:
		
		/** maximum number of iterations */
		//out_len=x_d		
		//w_len=x_d
		//grad_len=x_d
		//
		int32_t max_iter,wlen,sol_len,x_n,x_d,*sv,sv_len=0;
		float64_t* w,sol,grad;
		flaot64_t b,obj;
		SGVector<int32_t> train_labels;
		
		struct Options
		{	bool cg=false,lin_cg=false;
			int iter_max_Newton=20,cg_it=20;
			float prec=1e-6,cg_prec=1e-4;
		}
};
}
#endif

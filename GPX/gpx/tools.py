from gpkit import Variable


def variable_input(varname, force_constant=False, *varargs, **kwargs):
    '''parses a variable Input

	Typical Usage
	-------------
	inside of the setup function of building a model

	self.x = variable_input('x', **kwargs, 'x', 'min', 'time to complete')

	if 'x' were defined in the constructor, the

	Arguements
	----------
	varname : string
	 	the name of the variable
	**kwargs : dict
		the keyword arguments that should be passed from the variable constructor to find the variable

	Returns
	-------
	variable, constant, monomial

	'''
    # if the variable is not found in the input, create a variable
    if varname not in kwargs:
        return Variable(*varargs)

    # otherwise, just return the argument
    return kwargs[varname]


def constraints_from_string():
    '''constraints from string

	Creates a list of constraints from an input string

	'''
    # create the variables in the constraints
    # parse the constraints
    # return the constraints

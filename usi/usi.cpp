#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python.hpp>

namespace py = boost::python;

int main()
{
	Py_Initialize();

	py::object dlshogi_ns = py::import("dlshogi.usi").attr("__dict__");

	py::object dlshogi_usi_main = dlshogi_ns["main"];
	dlshogi_usi_main();

	Py_Finalize();

	return 0;
}
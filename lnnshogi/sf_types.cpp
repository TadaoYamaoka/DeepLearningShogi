#include "sf_types.h"

namespace Stockfish {

Value DrawValue[REPETITION_NB][COLOR_NB] =
{
    {  VALUE_ZERO     ,  VALUE_ZERO     }, // NOT_REPETITION
    {  VALUE_ZERO     ,  VALUE_ZERO     }, // REPETITION_DRAW
    {  VALUE_MATE     ,  VALUE_MATE     }, // REPETITION_WIN
    { -VALUE_MATE     , -VALUE_MATE     }, // REPETITION_LOSE
    {  VALUE_SUPERIOR ,  VALUE_SUPERIOR }, // REPETITION_SUPERIOR
    { -VALUE_SUPERIOR , -VALUE_SUPERIOR }, // REPETITION_INFERIOR
};

}  // namespace Stockfish

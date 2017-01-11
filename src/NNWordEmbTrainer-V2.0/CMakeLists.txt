
include_directories(
  model
)

add_executable(NNWordEmbTrainer-V2.0 NNWordEmbTrainer.cpp)
target_link_libraries(NNWordEmbTrainer-V2.0 ${LIBS})

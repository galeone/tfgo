module github.com/galeone/tfgo

go 1.15

require (
	github.com/tensorflow/tensorflow v2.4.1+incompatible
	google.golang.org/protobuf v1.25.0 // indirect
)

replace github.com/tensorflow/tensorflow => github.com/galeone/tensorflow v2.4.0-rc0.0.20210122160849-e898e11bfbb0+incompatible

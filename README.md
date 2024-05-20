# İnteraktif Yapay Zeka Modeli - İki sayının toplamı

2 sayısının toplamını tahmin eden, eğitilebilen javascript yapay zeka modeli.

## İnteraktif model oturumunu başlatma

```bash
$ yarn session

Initial model training complete
Enter "train" to provide new data, "test" to test the model, or "exit" to quit:

```

## Modeli eğitme

`train` komutunu verip, `<sayı 1>,<sayı 2>` şeklinde bir girdi verip ardından beklenen toplamı yazıp modeli eğitebilirsiniz. Örnek olarak aşağıdaki konsol çıktısını örnek alabilirsiniz.

```bash
Initial model training complete
Enter "train" to provide new data, "test" to test the model, or "exit" to quit: **train**
Enter two numbers separated by a comma for input (e.g., "9,10"): **1,2**
Enter the expected output for these inputs: 3
Retraining model with new data...
Model retraining complete
Enter "train" to provide new data, "test" to test the model, or "exit" to quit:
```

## Modeli test etme

`test` komutunu verip, `<sayı 1>,<sayı 2>` şeklinde bir girdi verip ardından modelden gelen çıktıyı görebilirsiniz. Örnek olarak aşağıdaki konsol çıktısını örnek alabilirsiniz.

```bash
Initial model training complete
Enter "train" to provide new data, "test" to test the model, or "exit" to quit: **test**
Enter two numbers separated by a comma to test (e.g., "9,10"): **1,2**
Tensor
     [[3.0031807],]
Enter "train" to provide new data, "test" to test the model, or "exit" to quit:
```

> `Enter "train" to provide new data, "test" to test the model, or "exit" to quit:` aşamasında `exit` yaparak oturumu durdurabilirsiniz.

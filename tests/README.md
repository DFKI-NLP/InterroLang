# pytest framework 
If you want to run a single test file, then:
```bash
cd test
pytest test_xxx.py
```

Otherwise, if you want to test all test files contained in the **test** folder, then:
```bash 
At root:

pytest tests --global --root
```
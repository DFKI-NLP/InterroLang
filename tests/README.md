# pytest framework 
If you want to run a single test file, then:
```bash
cd test
pytest tests/test_xxx.py --global --root
```

Otherwise, if you want to test all test files contained in the **test** folder, then:
```bash 
At root:

pytest tests --global --root
```
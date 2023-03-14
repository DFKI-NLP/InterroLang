# pytest framework 
If you want to test a single test file, then:
```bash
cd test
pytest -m test_xxx.py
```

Otherwise, if you want to test all test files contained in the **test** folder, then:
```bash 
At root:

pytest test/
```

However, paths in some files should be adapted, i.e. test_topk.py, in which neccessary changes are marked.
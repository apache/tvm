const tsParser = require('@typescript-eslint/parser');
const tsPlugin = require('@typescript-eslint/eslint-plugin');

module.exports = [
  {
    ignores :
            ['dist/**', 'debug/**', 'tvmjs_runtime_wasi.js', 'src/tvmjs_runtime_wasi.js', 'lib/**'],
  },
  {
    files : ['**/*.{js,mjs,cjs,ts,tsx}'],
    languageOptions : {
      ecmaVersion : 2018,
      sourceType : 'module',
      globals : {
        window : 'readonly',
        document : 'readonly',
        navigator : 'readonly',
        console : 'readonly',
        fetch : 'readonly',
        WebAssembly : 'readonly',
      },
    },
  },
  {
    files : ['src/**/*.ts', 'src/**/*.tsx'],
    languageOptions : {
      parser : tsParser,
      ecmaVersion : 2018,
      sourceType : 'module',
    },
    plugins : {
      '@typescript-eslint' : tsPlugin,
    },
    rules : {
      'require-jsdoc' : 0,
      '@typescript-eslint/no-explicit-any' : 0,
      '@typescript-eslint/no-empty-function' : 0,
      '@typescript-eslint/ban-types' : 'off',
    },
  },
  {
    files : ['tests/node/*.js', 'apps/node/*.js'],
    languageOptions : {
      globals : {
        require : 'readonly',
        module : 'readonly',
        process : 'readonly',
        __dirname : 'readonly',
      },
    },
  },
];

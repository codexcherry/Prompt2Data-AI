module.exports = {
    testEnvironment: 'node',
    testMatch: ['**/tests/**/*.test.js'],
    verbose: true,
    collectCoverageFrom: [
        'public/**/*.js',
        'server.js',
        '!**/node_modules/**'
    ],
    coverageDirectory: 'coverage',
    testTimeout: 30000
};

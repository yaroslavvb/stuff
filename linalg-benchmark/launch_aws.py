# Run linalg benchmark on AWS

import ncluster

parser.add_argument('--instances', default='p3.16xlarge, c5.18xlarge')
parser.add_argument('--ami', default="Deep Learning AMI (Amazon Linux) Version 13.0")

args = parser.parse_args()

set -x
set -e

option=$1
Name=$2
Region=$3
VPC=$4

if [[ "$option" == "create" ]]; then
aws --region $Region route53 create-hosted-zone --name $Name --vpc VPCRegion=$Region,VPCId=$VPC --caller-reference $Name.$(date "+%F-%T")
exit 0
elif [[ "$option" == "delete" ]]; then
HostedZoneId=$(aws --region $Region route53 list-hosted-zones --query "HostedZones[?Name == '$Name'].Id" --output text | sed 's/\/hostedzone\///g')
if [[ ! -z $HostedZoneId ]]; then
aws --region $Region route53 list-resource-record-sets \
  --hosted-zone-id $HostedZoneId |
jq -c '.ResourceRecordSets[]' |
while read -r resourcerecordset; do
  read -r name type <<<$(echo $(jq -r '.Name,.Type' <<<"$resourcerecordset"))
  if [ $type != "NS" -a $type != "SOA" ]; then
    aws --region $Region route53 change-resource-record-sets \
      --hosted-zone-id $HostedZoneId \
      --change-batch '{"Changes":[{"Action":"DELETE","ResourceRecordSet": '"$resourcerecordset"' }]}' \
      --output text --query 'ChangeInfo.Id'
  fi
done
    aws --region $Region route53 delete-hosted-zone --id $HostedZoneId
fi
exit 0
else
exit 1
fi
